
import os
import asyncio
import json
import urllib.request
import urllib.error

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import google
from rag_service import generate_policy_answer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

API_URL = "http://127.0.0.1:8000"


def api_request(path: str, method: str = "GET",
                payload: dict | None = None,
                access_token: str | None = None) -> dict:
    """Call the existing FastAPI backend."""
    headers = {"Content-Type": "application/json"}

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get(
                "detail", "Request failed"
            )
        except Exception:
            detail = "Request failed"
        return {"error": str(detail), "status_code": exc.code}
    except Exception as exc:
        print(f"Backend request failed: {type(exc).__name__}")
        return {"error": "The customer service backend is unavailable."}


class EcommerceVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly e-commerce customer service voice assistant. "
                "Speak clearly and concisely in English. "
                "For questions about returns, cancellations, shipping, delivery, "
                "or refunds, call search_commerce_policies and base your answer "
                "on its result. "
                "For order details, first verify the customer using "
                "verify_customer. Never disclose order information before "
                "successful verification. After verification, use "
                "get_my_orders or get_my_order_details. "
                "Never invent order details, tracking information, or policies. "
                "Never ask the customer to repeat or reveal their PIN after "
                "verification succeeds. Never speak or reveal access tokens. "
                "Do not cancel orders or create return requests in this step. "
                "If verification or lookup fails, explain the problem politely."
                "For cancellation requests, first use the prepare_order_cancellation tool. "
                "Explain the order and eligibility before asking for confirmation. "
                "Only call confirm_order_cancellation after the customer clearly says "
                "yes to cancelling that specific order. "
                "Never interpret uncertainty or an unrelated yes as cancellation confirmation. "
                "Never claim cancellation succeeded unless the backend confirms success. "
            )
        )
        self.access_token = None
        self.customer_id = None

    @function_tool()
    async def search_commerce_policies(
        self,
        context: RunContext,
        question: str,
    ) -> dict:
        """Search official policies for returns, cancellations, shipping,
        delivery, and refunds."""
        try:
            return await asyncio.to_thread(generate_policy_answer, question)
        except Exception as exc:
            print(f"Policy tool error: {type(exc).__name__}")
            return {
                "answer": (
                    "I couldn't retrieve the policy right now. "
                    "Please contact customer support."
                ),
                "sources": [],
            }

    @function_tool()
    async def verify_customer(
        self,
        context: RunContext,
        customer_id: str,
        verification_pin: str,
    ) -> dict:
        """Verify a customer using their customer ID and four-digit PIN.
        Call this before accessing any order information."""
        result = await asyncio.to_thread(
            api_request,
            "/api/customers/verify",
            "POST",
            {
                "customer_id": customer_id.strip().upper(),
                "verification_pin": verification_pin.strip(),
            },
        )

        if result.get("error") or not result.get("verified"):
            self.access_token = None
            self.customer_id = None
            return {
                "verified": False,
                "message": (
                    "Verification failed. Please check your customer ID and PIN."
                ),
            }

        # Keep the credential in this agent instance, not in the tool response.
        self.access_token = result["access_token"]
        self.customer_id = result["customer_id"]

        return {
            "verified": True,
            "customer_name": result["customer_name"],
            "message": (
                "Customer verified successfully. You may now look up "
                "this customer's orders."
            ),
        }

    @function_tool()
    async def get_my_orders(self, context: RunContext) -> dict:
        """List the verified customer's orders. Requires successful verification."""
        if not self.access_token or not self.customer_id:
            return {
                "success": False,
                "message": "Please verify your identity before looking up orders.",
            }

        result = await asyncio.to_thread(
            api_request,
            "/api/customers/me/orders",
            "GET",
            None,
            self.access_token,
        )

        if result.get("status_code") == 401:
            self.access_token = None
            self.customer_id = None
            return {
                "success": False,
                "message": "Your verification session expired. Please verify again.",
            }

        if result.get("error"):
            return {
                "success": False,
                "message": "I couldn't retrieve your orders right now.",
            }

        return {"success": True, "orders": result.get("orders", [])}


    @function_tool()
    async def prepare_order_cancellation(
        self,
        context: RunContext,
        order_id: str,
    ) -> dict:
        """Check a verified customer's order before requesting cancellation.
        Call this before asking the customer to confirm cancellation."""
        if not self.access_token or not self.customer_id:
            return {
                "success": False,
                "message": "Please verify your identity before cancelling an order.",
            }

        result = await asyncio.to_thread(
            api_request,
            f"/api/orders/{order_id.strip().upper()}",
            "GET",
            None,
            self.access_token,
        )

        if result.get("error"):
            return {
                "success": False,
                "message": result["error"],
            }

        eligible = (
            result.get("status") == "processing"
            and result.get("shipment_status") == "not_shipped"
        )

        if not eligible:
            return {
                "success": False,
                "order_id": result.get("order_id"),
                "status": result.get("status"),
                "shipment_status": result.get("shipment_status"),
                "message": "This order is not eligible for cancellation.",
            }

        return {
            "success": True,
            "order_id": result["order_id"],
            "product": result["product"],
            "status": result["status"],
            "shipment_status": result["shipment_status"],
            "message": (
                "This order is eligible for cancellation. "
                "Ask the customer for explicit confirmation before proceeding."
            ),
        }

    @function_tool()
    async def confirm_order_cancellation(
        self,
        context: RunContext,
        order_id: str,
        customer_confirmation: str,
    ) -> dict:
        """Execute cancellation only after the customer explicitly confirms
        cancelling this specific order. The confirmation must be an exact
        affirmative response such as 'yes, cancel it'."""
        if not self.access_token or not self.customer_id:
            return {
                "success": False,
                "message": "Please verify your identity before cancelling an order.",
            }

        confirmation = customer_confirmation.strip().lower()
        allowed_confirmations = {
            "yes, cancel it",
            "yes cancel it",
            "yes, cancel the order",
            "yes cancel the order",
            "i confirm cancellation",
            "i confirm cancelling the order",
        }

        if confirmation not in allowed_confirmations:
            return {
                "success": False,
                "message": (
                    "Cancellation was not performed. Ask the customer to clearly "
                    "confirm cancellation of this specific order."
                ),
            }

        order_id = order_id.strip().upper()

        # Recheck eligibility immediately before the transaction.
        check = await asyncio.to_thread(
            api_request,
            f"/api/orders/{order_id}",
            "GET",
            None,
            self.access_token,
        )

        if check.get("error"):
            return {
                "success": False,
                "message": check["error"],
            }

        if (
            check.get("status") != "processing"
            or check.get("shipment_status") != "not_shipped"
        ):
            return {
                "success": False,
                "message": "This order is no longer eligible for cancellation.",
            }

        import uuid

        idempotency_key = str(uuid.uuid4())

        result = await asyncio.to_thread(
            api_request,
            f"/api/orders/{order_id}/cancel",
            "POST",
            {
                "confirmation": True,
                "idempotency_key": idempotency_key,
            },
            self.access_token,
        )

        if result.get("error"):
            return {
                "success": False,
                "message": result["error"],
            }

        return {
            "success": True,
            "order_id": result["order_id"],
            "status": result["status"],
            "message": "The backend confirmed that the order was cancelled.",
        }
    

    @function_tool()
    async def get_my_order_details(
        self,
        context: RunContext,
        order_id: str,
    ) -> dict:
        """Retrieve details of one order belonging to the verified customer."""
        if not self.access_token or not self.customer_id:
            return {
                "success": False,
                "message": "Please verify your identity before looking up an order.",
            }

        result = await asyncio.to_thread(
            api_request,
            f"/api/orders/{order_id.strip().upper()}",
            "GET",
            None,
            self.access_token,
        )

        if result.get("status_code") == 401:
            self.access_token = None
            self.customer_id = None
            return {
                "success": False,
                "message": "Your verification session expired. Please verify again.",
            }

        if result.get("status_code") == 404:
            return {
                "success": False,
                "message": "I couldn't find that order under your account.",
            }

        if result.get("error"):
            return {
                "success": False,
                "message": "I couldn't retrieve that order right now.",
            }

        # Return only the fields needed for a spoken status response.
        return {
            "success": True,
            "order": {
                "order_id": result["order_id"],
                "product": result["product"],
                "status": result["status"],
                "shipment_status": result["shipment_status"],
                "order_date": result["order_date"],
                "delivered_date": result["delivered_date"],
            },
        }


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-3.8-live",
            voice="Aoede",
        )
    )

    await ctx.connect()
    await session.start(room=ctx.room, agent=EcommerceVoiceAgent())
    await session.generate_reply(
        instructions="Greet the customer and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
