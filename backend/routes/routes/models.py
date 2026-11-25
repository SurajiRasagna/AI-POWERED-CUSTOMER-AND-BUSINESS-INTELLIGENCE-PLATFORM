"""Database models: users and prediction audit log."""

from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from database import db

ROLES = ("admin", "analyst", "viewer")


def utcnow():
    """Naive UTC timestamp for database storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="analyst")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    last_login_at = db.Column(db.DateTime)

    predictions = db.relationship(
        "PredictionLog",
        back_populates="user",
        lazy="dynamic",
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "admin"

    @property
    def initials(self):
        parts = [p for p in (self.name or "").split() if p]
        if not parts:
            return (self.email or "?")[0].upper()
        if len(parts) == 1:
            return parts[0][0].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": (
                self.last_login_at.isoformat() if self.last_login_at else None
            ),
        }


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(64), nullable=False, index=True)
    forecast_month = db.Column(db.Date, nullable=False)
    predicted_quantity = db.Column(db.Float, nullable=False)
    signal = db.Column(db.String(32))
    safety_stock = db.Column(db.Float, default=0.0)
    reorder_point = db.Column(db.Float, default=0.0)
    source = db.Column(db.String(20), nullable=False, default="single")
    created_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    user = db.relationship("User", back_populates="predictions")

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "forecast_month": (
                self.forecast_month.strftime("%Y-%m")
                if self.forecast_month
                else None
            ),
            "predicted_quantity": round(self.predicted_quantity, 2),
            "signal": self.signal,
            "safety_stock": round(self.safety_stock or 0.0, 2),
            "reorder_point": round(self.reorder_point or 0.0, 2),
            "source": self.source,
            "created_by": self.user.name if self.user else "system",
            "created_at": (
                self.created_at.strftime("%d %b %Y %H:%M")
                if self.created_at
                else None
            ),
        }
