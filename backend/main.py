
import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from backend.database import get_connection, initialize_database

from backend.rag_service import index_policies, retrieve_policies, generate_policy_answer

import os

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

from backend.admin_api import router as admin_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI(
    title="E-commerce Voice Customer Service API",
    version="0.3.0",
)

load_dotenv(PROJECT_ROOT / ".env")

app.include_router(admin_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer(auto_error=False)


def now():
    return datetime.now(timezone.utc)


def load_json(filename):
    with (DATA_DIR / filename).open(encoding="utf-8") as f:
        return json.load(f)


customers = load_json("customers.json")


def audit(conn, event_type, customer_id, order_id, outcome):
    conn.execute(
        """INSERT INTO audit_log
           (event_type, customer_id, order_id, outcome, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (event_type, customer_id, order_id, outcome, now().isoformat()),
    )


def initialize_commerce_data():
    initialize_database()
    seed_orders = load_json("orders.json")

    with get_connection() as conn:
        for order in seed_orders:
            conn.execute(
                """INSERT OR IGNORE INTO orders
                   (order_id, customer_id, product, amount, status,
                    order_date, delivered_date, shipment_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order["order_id"], order["customer_id"],
                    order["product"], order["amount"], order["status"],
                    order["order_date"], order["delivered_date"],
                    order["shipment_status"],
                ),
            )


@app.on_event("startup")
def startup():
    initialize_commerce_data()
    index_policies()


class VerifyRequest(BaseModel):
    customer_id: str
    verification_pin: str = Field(min_length=4, max_length=4)


class ReturnRequest(BaseModel):
    order_id: str
    reason: str = Field(min_length=3, max_length=300)
    confirmation: bool
    idempotency_key: str = Field(min_length=8, max_length=100)


class CancelRequest(BaseModel):
    confirmation: bool
    idempotency_key: str = Field(min_length=8, max_length=100)


def current_customer(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Verification required")

    token_hash = hashlib.sha256(
        credentials.credentials.encode()
    ).hexdigest()

    with get_connection() as conn:
        session = conn.execute(
            """SELECT customer_id, expires_at
               FROM verification_sessions WHERE token_hash = ?""",
            (token_hash,),
        ).fetchone()

    if not session or datetime.fromisoformat(session["expires_at"]) <= now():
        raise HTTPException(status_code=401, detail="Session invalid or expired")

    return session["customer_id"]


def get_owned_order(conn, order_id, customer_id):
    order = conn.execute(
        "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
        (order_id, customer_id),
    ).fetchone()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return order


def idempotent_response(conn, key, operation, request_hash):
    record = conn.execute(
        "SELECT * FROM idempotency_records WHERE idempotency_key = ?",
        (key,),
    ).fetchone()

    if not record:
        return None

    if record["operation"] != operation or record["request_hash"] != request_hash:
        raise HTTPException(
            status_code=409,
            detail="Idempotency key reused with different request",
        )

    return json.loads(record["response_json"])


def save_idempotent_response(conn, key, operation, request_hash, response):
    conn.execute(
        """INSERT INTO idempotency_records
           (idempotency_key, operation, request_hash, response_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            key, operation, request_hash, json.dumps(response),
            now().isoformat(),
        ),
    )


@app.get("/")
def root():
    return {"message": "E-commerce Voice Customer Service API", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "healthy", "timestamp": now().isoformat()}


@app.post("/api/customers/verify")
def verify_customer(request: VerifyRequest):
    customer = next(
        (c for c in customers if c["customer_id"] == request.customer_id),
        None,
    )

    if not customer or not secrets.compare_digest(
        customer["verification_pin"], request.verification_pin
    ):
        with get_connection() as conn:
            audit(conn, "customer_verification", request.customer_id, None, "failed")
        raise HTTPException(status_code=401, detail="Customer verification failed")

    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    created = now()
    expires = created + timedelta(minutes=15)

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO verification_sessions
               (token_hash, customer_id, created_at, expires_at)
               VALUES (?, ?, ?, ?)""",
            (token_hash, customer["customer_id"], created.isoformat(), expires.isoformat()),
        )
        audit(conn, "customer_verification", customer["customer_id"], None, "success")

    return {
        "verified": True,
        "customer_id": customer["customer_id"],
        "customer_name": customer["name"],
        "access_token": token,
        "expires_at": expires.isoformat(),
    }


@app.get("/api/customers/me/orders")
def order_history(customer_id=Depends(current_customer)):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT order_id, product, amount, status, order_date
               FROM orders WHERE customer_id = ? ORDER BY order_date DESC""",
            (customer_id,),
        ).fetchall()

    return {"customer_id": customer_id, "orders": [dict(row) for row in rows]}


@app.get("/api/orders/{order_id}")
def order_status(order_id: str, customer_id=Depends(current_customer)):
    with get_connection() as conn:
        order = get_owned_order(conn, order_id, customer_id)
    return dict(order)


@app.post("/api/orders/{order_id}/cancel")
def cancel_order(
    order_id: str,
    request: CancelRequest,
    customer_id=Depends(current_customer),
):
    if not request.confirmation:
        raise HTTPException(status_code=400, detail="Explicit confirmation required")

    operation = "cancel_order"
    request_hash = hashlib.sha256(
        f"{customer_id}|{order_id}|{request.confirmation}".encode()
    ).hexdigest()

    with get_connection() as conn:
        prior = idempotent_response(
            conn, request.idempotency_key, operation, request_hash
        )
        if prior is not None:
            return prior

        order = get_owned_order(conn, order_id, customer_id)

        if order["status"] != "processing" or order["shipment_status"] != "not_shipped":
            raise HTTPException(status_code=409, detail="Order is not eligible for cancellation")

        conn.execute(
            """UPDATE orders SET status = 'cancelled'
               WHERE order_id = ? AND customer_id = ?
                 AND status = 'processing' AND shipment_status = 'not_shipped'""",
            (order_id, customer_id),
        )

        response = {"success": True, "order_id": order_id, "status": "cancelled"}
        save_idempotent_response(
            conn, request.idempotency_key, operation, request_hash, response
        )
        audit(conn, operation, customer_id, order_id, "success")

    return response


@app.post("/api/returns")
def create_return(
    request: ReturnRequest,
    customer_id=Depends(current_customer),
):
    if not request.confirmation:
        raise HTTPException(status_code=400, detail="Explicit confirmation required")

    operation = "create_return"
    request_hash = hashlib.sha256(
        f"{customer_id}|{request.order_id}|{request.reason}".encode()
    ).hexdigest()

    with get_connection() as conn:
        prior = idempotent_response(
            conn, request.idempotency_key, operation, request_hash
        )
        if prior is not None:
            return prior

        order = get_owned_order(conn, request.order_id, customer_id)

        if order["status"] != "delivered" or not order["delivered_date"]:
            raise HTTPException(status_code=409, detail="Order is not eligible for return")

        delivered = datetime.strptime(order["delivered_date"], "%Y-%m-%d").date()
        if (now().date() - delivered).days > 10:
            raise HTTPException(status_code=409, detail="Return window has expired")

        existing = conn.execute(
            """SELECT return_id FROM return_requests
               WHERE order_id = ? AND status = 'requested'""",
            (request.order_id,),
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Return already requested")

        return_id = "RET-" + uuid4().hex[:10].upper()
        conn.execute(
            """INSERT INTO return_requests
               (return_id, order_id, customer_id, reason, status, created_at, idempotency_key)
               VALUES (?, ?, ?, ?, 'requested', ?, ?)""",
            (
                return_id, request.order_id, customer_id, request.reason,
                now().isoformat(), request.idempotency_key,
            ),
        )

        response = {
            "success": True,
            "return_id": return_id,
            "order_id": request.order_id,
            "status": "requested",
        }
        save_idempotent_response(
            conn, request.idempotency_key, operation, request_hash, response
        )
        audit(conn, operation, customer_id, request.order_id, "success")

    return response


@app.get("/api/returns")
def list_returns(customer_id=Depends(current_customer)):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT return_id, order_id, reason, status, created_at
               FROM return_requests WHERE customer_id = ?""",
            (customer_id,),
        ).fetchall()
    return {"returns": [dict(row) for row in rows]}


class PolicyQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.post("/api/knowledge/search")
def search_knowledge(request: PolicyQuestion):
    try:
        matches = retrieve_policies(request.question, top_k=3)
        return {
            "question": request.question,
            "matches": matches,
        }
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Policy retrieval is temporarily unavailable",
        )


@app.post("/api/knowledge/answer")
def answer_knowledge(request: PolicyQuestion):
    try:
        return generate_policy_answer(request.question)
    except Exception as exc:
        print(f"Policy answering failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Policy answering failed: {type(exc).__name__}: {exc}",
        )


@app.post("/api/voice/token")
def create_voice_token():
    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([livekit_url, api_key, api_secret]):
        raise HTTPException(
            status_code=503,
            detail="LiveKit configuration is missing",
        )

    room_name = "ecommerce-voice-demo"
    participant_identity = f"web-{secrets.token_hex(6)}"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("E-commerce Customer")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(minutes=15))
    )

    return {
        "server_url": livekit_url,
        "participant_token": token.to_jwt(),
        "room_name": room_name,
    }