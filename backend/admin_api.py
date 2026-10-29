"""Admin CRUD API for the local e-commerce voice POC.

Mount from the existing FastAPI app:
    from backend.admin_api import router as admin_router
    app.include_router(admin_router)

Set ADMIN_API_KEY in .env before starting the API. This is a local POC admin
surface; do not expose it publicly without stronger authentication/authorization.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.database import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CUSTOMERS_FILE = DATA_DIR / "customers.json"
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "admin.html"

router = APIRouter(tags=["admin-crud"])


def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    expected = os.getenv("ADMIN_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY is not configured in the backend .env file.")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="Invalid admin API key.")
    return True


def read_customers() -> list[dict[str, Any]]:
    if not CUSTOMERS_FILE.exists():
        raise HTTPException(status_code=500, detail="customers.json was not found.")
    with CUSTOMERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_customers(items: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="customers_", suffix=".json", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_name, CUSTOMERS_FILE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def refresh_customer_cache(items: list[dict[str, Any]]) -> None:
    # Update the API module's in-memory customer list, if it has already loaded it.
    for module in list(sys.modules.values()):
        try:
            current = getattr(module, "customers", None)
            module_data_dir = getattr(module, "DATA_DIR", None)
            if isinstance(current, list) and module_data_dir is not None:
                candidate = Path(module_data_dir) / "customers.json"
                if candidate.resolve() == CUSTOMERS_FILE.resolve():
                    current[:] = items
        except Exception:
            continue


def safe_customer(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k != "verification_pin"}


class CustomerIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=60)
    name: str = Field(min_length=1, max_length=120)
    verification_pin: str | None = Field(default=None, pattern=r"^\d{4}$")
    email: str = ""
    phone: str = ""


class OrderIn(BaseModel):
    order_id: str = Field(min_length=1, max_length=80)
    customer_id: str = Field(min_length=1, max_length=60)
    product: str = Field(min_length=1, max_length=200)
    amount: float = Field(ge=0)
    status: str = Field(default="processing", min_length=1, max_length=40)
    order_date: str = Field(min_length=4, max_length=40)
    delivered_date: str | None = None
    shipment_status: str = Field(default="not_shipped", min_length=1, max_length=40)


@router.get("/admin", include_in_schema=False)
def admin_page():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Admin frontend file not found. Place it at frontend/admin.html.")
    return FileResponse(FRONTEND_FILE, media_type="text/html")


@router.get("/api/admin/customers")
def list_customers(_: bool = Depends(require_admin)):
    return [safe_customer(c) for c in read_customers()]


@router.post("/api/admin/customers")
def create_customer(request: CustomerIn, _: bool = Depends(require_admin)):
    item = request.model_dump()
    item["customer_id"] = item["customer_id"].strip().upper()
    item["name"] = item["name"].strip()
    if not item.get("verification_pin") or len(item["verification_pin"]) != 4:
        raise HTTPException(status_code=422, detail="A four-digit verification PIN is required.")
    customers = read_customers()
    if any(c.get("customer_id", "").upper() == item["customer_id"] for c in customers):
        raise HTTPException(status_code=409, detail="Customer ID already exists.")
    customers.append(item)
    write_customers(customers)
    refresh_customer_cache(customers)
    return {"success": True, "customer": safe_customer(item)}


@router.put("/api/admin/customers/{customer_id}")
def update_customer(customer_id: str, request: CustomerIn, _: bool = Depends(require_admin)):
    if request.customer_id.strip().upper() != customer_id.strip().upper():
        raise HTTPException(status_code=400, detail="Customer ID cannot be changed.")
    customers = read_customers()
    for i, item in enumerate(customers):
        if item.get("customer_id", "").upper() == customer_id.upper():
            updated = dict(item)
            updated["name"] = request.name.strip()
            updated["email"] = request.email
            updated["phone"] = request.phone
            if request.verification_pin:
                updated["verification_pin"] = request.verification_pin
            customers[i] = updated
            write_customers(customers)
            refresh_customer_cache(customers)
            return {"success": True, "customer": safe_customer(updated)}
    raise HTTPException(status_code=404, detail="Customer not found.")


@router.delete("/api/admin/customers/{customer_id}")
def delete_customer(customer_id: str, _: bool = Depends(require_admin)):
    customers = read_customers()
    match = next((c for c in customers if c.get("customer_id", "").upper() == customer_id.upper()), None)
    if not match:
        raise HTTPException(status_code=404, detail="Customer not found.")
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM orders WHERE customer_id = ?", (customer_id,)).fetchone()["n"]
        if count:
            raise HTTPException(status_code=409, detail="This customer has orders. Delete or reassign their orders before deleting the customer.")
    customers = [c for c in customers if c.get("customer_id", "").upper() != customer_id.upper()]
    write_customers(customers)
    refresh_customer_cache(customers)
    return {"success": True}


@router.get("/api/admin/orders")
def list_orders(_: bool = Depends(require_admin)):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT order_id, customer_id, product, amount, status,
                      order_date, delivered_date, shipment_status
               FROM orders ORDER BY order_date DESC, order_id"""
        ).fetchall()
    return [dict(row) for row in rows]


@router.post("/api/admin/orders")
def create_order(request: OrderIn, _: bool = Depends(require_admin)):
    data = request.model_dump()
    data["order_id"] = data["order_id"].strip().upper()
    data["customer_id"] = data["customer_id"].strip().upper()
    if not any(c.get("customer_id", "").upper() == data["customer_id"] for c in read_customers()):
        raise HTTPException(status_code=422, detail="Customer ID does not exist.")
    with get_connection() as conn:
        exists = conn.execute("SELECT 1 FROM orders WHERE order_id = ?", (data["order_id"],)).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Order ID already exists.")
        conn.execute(
            """INSERT INTO orders
               (order_id, customer_id, product, amount, status, order_date, delivered_date, shipment_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (data["order_id"], data["customer_id"], data["product"].strip(), data["amount"],
             data["status"], data["order_date"], data.get("delivered_date"), data["shipment_status"]),
        )
    return {"success": True, "order": data}


@router.put("/api/admin/orders/{order_id}")
def update_order(order_id: str, request: OrderIn, _: bool = Depends(require_admin)):
    data = request.model_dump()
    if data["order_id"].strip().upper() != order_id.strip().upper():
        raise HTTPException(status_code=400, detail="Order ID cannot be changed.")
    data["customer_id"] = data["customer_id"].strip().upper()
    if not any(c.get("customer_id", "").upper() == data["customer_id"] for c in read_customers()):
        raise HTTPException(status_code=422, detail="Customer ID does not exist.")
    with get_connection() as conn:
        cur = conn.execute(
            """UPDATE orders SET customer_id=?, product=?, amount=?, status=?,
                      order_date=?, delivered_date=?, shipment_status=?
               WHERE order_id=?""",
            (data["customer_id"], data["product"].strip(), data["amount"], data["status"],
             data["order_date"], data.get("delivered_date"), data["shipment_status"], order_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Order not found.")
    return {"success": True, "order": data}


@router.delete("/api/admin/orders/{order_id}")
def delete_order(order_id: str, _: bool = Depends(require_admin)):
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM orders WHERE order_id = ?", (order_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Order not found.")
    return {"success": True}
