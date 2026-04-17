"""Mock tool handlers for the customer support example.

Backed by a small in-memory order database keyed to the order IDs used in
``scenarios.yaml``. Each handler returns the shape described in
``components/tools/v1.yaml`` so the agent can reason about realistic
results without a real backend.

Order dates are generated relative to "today" so that return-window logic
(within 30 days vs. 45 days ago) stays correct whenever the example runs.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

_TODAY = datetime.now(timezone.utc).date()


def _d(offset_days: int) -> str:
    return (_TODAY - timedelta(days=offset_days)).isoformat()


# Order ID → record. IDs and statuses are aligned with examples/customer_support/scenarios.yaml.
_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-78234": {
        "order_id": "ORD-78234",
        "items": ["Wireless Headphones"],
        "status": "shipped",
        "order_date": _d(5),
        "total": 129.99,
        "payment_method_last4": "4242",
    },
    "ORD-45123": {
        "order_id": "ORD-45123",
        "items": ["Running Shoes"],
        "status": "delivered",
        "order_date": _d(10),
        "total": 89.00,
        "payment_method_last4": "1111",
    },
    "ORD-91001": {
        "order_id": "ORD-91001",
        "items": ["T-Shirt"],
        "status": "processing",
        "order_date": _d(1),
        "total": 25.00,
        "payment_method_last4": "5555",
    },
    "ORD-33201": {
        "order_id": "ORD-33201",
        "items": ["Coffee Maker"],
        "status": "shipped",
        "order_date": _d(3),
        "total": 79.50,
        "payment_method_last4": "9999",
    },
    "ORD-20100": {
        "order_id": "ORD-20100",
        "items": ["Desk Lamp"],
        "status": "delivered",
        "order_date": _d(45),
        "total": 59.00,
        "payment_method_last4": "2222",
    },
    "ORD-55400": {
        "order_id": "ORD-55400",
        "items": ["Laptop Stand", "External Monitor"],
        "status": "processing",
        "order_date": _d(7),
        "total": 800.00,
        "payment_method_last4": "7777",
    },
    "ORD-67890": {
        "order_id": "ORD-67890",
        "items": ["Backpack"],
        "status": "shipped",
        "order_date": _d(8),
        "total": 110.00,
        "payment_method_last4": "3333",
    },
    "ORD-11200": {
        "order_id": "ORD-11200",
        "items": ["Bluetooth Speaker"],
        "status": "delivered",
        "order_date": _d(14),
        "total": 65.00,
        "payment_method_last4": "8888",
    },
    "ORD-44100": {
        "order_id": "ORD-44100",
        "items": ["Phone Case"],
        "status": "shipped",
        "order_date": _d(2),
        "total": 24.99,
        "payment_method_last4": "6161",
    },
}


def _tracking_for(order: dict[str, Any]) -> dict[str, Any]:
    """Build a plausible tracking payload for a shipped/delivered order."""
    if order["status"] == "delivered":
        return {
            "carrier": "FedEx",
            "tracking_number": f"FDX{order['order_id'][-5:]}",
            "current_status": "delivered",
            "estimated_delivery": order["order_date"],
            "last_update": f"{order['order_date']}T18:00:00Z",
        }
    return {
        "carrier": "FedEx",
        "tracking_number": f"FDX{order['order_id'][-5:]}",
        "current_status": "in_transit",
        "estimated_delivery": _d(-2),
        "last_update": f"{_TODAY.isoformat()}T09:00:00Z",
    }


def lookup_order(order_id: str) -> dict[str, Any]:
    """Return the order record or a structured error."""
    order = _ORDERS.get(order_id)
    if order is None:
        return {
            "error": {
                "code": "order_not_found",
                "message": f"No order with id {order_id!r} exists.",
                "retryable": False,
            }
        }
    return dict(order)


def get_tracking_status(order_id: str) -> dict[str, Any]:
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": {"code": "order_not_found", "retryable": False}}
    if order["status"] in ("pending", "processing"):
        return {"error": {"code": "not_shipped", "retryable": False}}
    return _tracking_for(order)


def initiate_return(order_id: str, reason: str) -> dict[str, Any]:
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": {"code": "order_not_found", "retryable": False}}
    order_date = date.fromisoformat(order["order_date"])
    days = (_TODAY - order_date).days
    if days > 30:
        return {
            "error": {
                "code": "outside_window",
                "message": f"Order is {days} days old (limit 30).",
                "retryable": False,
                "reason_provided": reason,
            }
        }
    return {
        "return_id": f"RET-{uuid.uuid4().hex[:8].upper()}",
        "return_label_url": f"https://shopfast.example/returns/{order_id}.pdf",
        "refund_amount": order["total"],
        "refund_timeline": "5-7 business days after we receive the item",
    }


def cancel_order(order_id: str, reason: str) -> dict[str, Any]:
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": {"code": "order_not_found", "retryable": False}}
    if order["status"] in ("shipped", "delivered"):
        return {
            "error": {
                "code": "already_shipped",
                "message": "Cannot cancel — order has shipped.",
                "retryable": False,
                "reason_provided": reason,
            }
        }
    return {
        "cancellation_id": f"CAN-{uuid.uuid4().hex[:8].upper()}",
        "refund_amount": order["total"],
        "refund_timeline": "3-5 business days",
    }


def escalate_to_human(
    reason: str,
    priority: str,
    context_summary: str,
) -> dict[str, Any]:
    wait_by_priority = {"urgent": 2, "high": 8, "normal": 25}
    return {
        "transfer_id": f"TRN-{uuid.uuid4().hex[:8].upper()}",
        "estimated_wait_time": wait_by_priority.get(priority, 25),
        "queue_position": 3 if priority == "urgent" else 12,
        "accepted_reason": reason,
        "context_summary_received": context_summary,
    }


def get_handlers() -> dict[str, Any]:
    """Return the handler map expected by AgentRunner / ImprovementLoop."""
    return {
        "lookup_order": lookup_order,
        "get_tracking_status": get_tracking_status,
        "initiate_return": initiate_return,
        "cancel_order": cancel_order,
        "escalate_to_human": escalate_to_human,
    }
