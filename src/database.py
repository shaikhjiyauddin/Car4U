"""
database.py — lightweight SQLite backend for Car4U's Marketplace feature.

Handles:
  - user accounts (signup / login, passwords hashed with salted PBKDF2)
  - car listings submitted by users (the "owner" of a listing is a real account)
  - direct messages between a buyer and a listing's owner (no phone numbers involved)

This is intentionally dependency-free (uses only Python's built-in sqlite3 and
hashlib) so it doesn't add anything new to requirements.txt.
"""
import sqlite3
import hashlib
import secrets
import os
import io
import base64
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'car4u.db')

MAX_PHOTO_DIMENSION = 800  # px, longest side
PHOTO_JPEG_QUALITY = 70


def encode_photo(file_bytes):
    """Resize and compress an uploaded image into a small base64 JPEG string
    suitable for storing directly in SQLite. Returns None if Pillow isn't
    available or the bytes aren't a valid image, so photo upload degrades
    gracefully instead of crashing listing creation."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert('RGB')
        img.thumbnail((MAX_PHOTO_DIMENSION, MAX_PHOTO_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=PHOTO_JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return None


def get_connection(db_path=None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=None):
    """Create tables if they don't already exist. Safe to call on every app startup."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            price REAL NOT NULL,
            fuel_type TEXT,
            transmission TEXT,
            kilometers_driven INTEGER,
            location TEXT,
            description TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (listing_id) REFERENCES listings(id),
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (receiver_id) REFERENCES users(id)
        );
    """)
    conn.commit()

    # Track which listings have already been merged into the training CSV.
    # Added via ALTER TABLE (not the CREATE above) so it also applies to
    # databases created before this feature existed.
    try:
        conn.execute("ALTER TABLE listings ADD COLUMN exported_at TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # A listing's photo, stored as a base64-encoded, resized JPEG string.
    try:
        conn.execute("ALTER TABLE listings ADD COLUMN photo TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    conn.close()


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000).hex()
    return pwd_hash, salt


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------- Users ----------

def create_user(name, email, password, db_path=None):
    """Returns (success: bool, message: str)."""
    if not name.strip() or not email.strip() or not password:
        return False, "Name, email, and password are all required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    conn = get_connection(db_path)
    pwd_hash, salt = _hash_password(password)
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), email.lower().strip(), pwd_hash, salt, _now())
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."
    finally:
        conn.close()


def verify_login(email, password, db_path=None):
    """Returns a dict {id, name, email} on success, or None on failure."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    if row is None:
        return None
    pwd_hash, _ = _hash_password(password, row['salt'])
    if pwd_hash == row['password_hash']:
        return {'id': row['id'], 'name': row['name'], 'email': row['email']}
    return None


def get_user_by_id(user_id, db_path=None):
    conn = get_connection(db_path)
    row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- Listings ----------

def add_listing(owner_id, brand, model, year, price, fuel_type, transmission,
                 kilometers_driven, location, description, db_path=None, photo=None):
    conn = get_connection(db_path)
    cur = conn.execute(
        """INSERT INTO listings
           (owner_id, brand, model, year, price, fuel_type, transmission,
            kilometers_driven, location, description, created_at, photo)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (owner_id, brand.strip(), model.strip(), year, price, fuel_type, transmission,
         kilometers_driven, location, description, _now(), photo)
    )
    conn.commit()
    listing_id = cur.lastrowid
    conn.close()
    return listing_id


def get_all_listings(db_path=None):
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT listings.*, users.name AS owner_name
        FROM listings JOIN users ON listings.owner_id = users.id
        ORDER BY listings.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_listing(listing_id, db_path=None):
    conn = get_connection(db_path)
    row = conn.execute("""
        SELECT listings.*, users.name AS owner_name
        FROM listings JOIN users ON listings.owner_id = users.id
        WHERE listings.id = ?
    """, (listing_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_listings_by_owner(owner_id, db_path=None):
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM listings WHERE owner_id = ? ORDER BY created_at DESC", (owner_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unexported_listings(db_path=None):
    """Listings that haven't been merged into the training CSV yet."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM listings WHERE exported_at IS NULL ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_listings_exported(listing_ids, db_path=None):
    """Flag listings as merged into the training CSV so they aren't exported twice."""
    if not listing_ids:
        return
    conn = get_connection(db_path)
    conn.executemany(
        "UPDATE listings SET exported_at = ? WHERE id = ?",
        [(_now(), lid) for lid in listing_ids]
    )
    conn.commit()
    conn.close()


# ---------- Messages ----------

def send_message(listing_id, sender_id, receiver_id, body, db_path=None):
    if not body.strip():
        return False
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO messages (listing_id, sender_id, receiver_id, body, created_at) VALUES (?, ?, ?, ?, ?)",
        (listing_id, sender_id, receiver_id, body.strip(), _now())
    )
    conn.commit()
    conn.close()
    return True


def get_conversation(listing_id, user_a_id, user_b_id, db_path=None):
    """All messages between two users about one listing, oldest first."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT messages.*, users.name AS sender_name
        FROM messages JOIN users ON messages.sender_id = users.id
        WHERE listing_id = ?
          AND ((sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?))
        ORDER BY messages.created_at ASC
    """, (listing_id, user_a_id, user_b_id, user_b_id, user_a_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_conversations(user_id, db_path=None):
    """Distinct (listing_id, other_user_id) pairs this user has exchanged messages in,
    enriched with listing and other-user display info, most recent first."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT
            m.listing_id,
            CASE WHEN m.sender_id = ? THEN m.receiver_id ELSE m.sender_id END AS other_user_id,
            MAX(m.created_at) AS last_message_at
        FROM messages m
        WHERE m.sender_id = ? OR m.receiver_id = ?
        GROUP BY m.listing_id, other_user_id
        ORDER BY last_message_at DESC
    """, (user_id, user_id, user_id)).fetchall()
    conn.close()

    results = []
    for r in rows:
        listing = get_listing(r['listing_id'], db_path=db_path)
        other_user = get_user_by_id(r['other_user_id'], db_path=db_path)
        if listing and other_user:
            results.append({
                'listing_id': r['listing_id'],
                'listing_title': f"{listing['brand']} {listing['model']} ({listing['year']})",
                'other_user_id': other_user['id'],
                'other_user_name': other_user['name'],
                'last_message_at': r['last_message_at'],
            })
    return results


# ---------- Live data pipeline stats (for the Analytics dashboard) ----------

def get_pipeline_stats(db_path=None):
    """Snapshot of the live data pipeline: how much real, user-generated data
    exists right now, and how much of it has already fed into model training."""
    conn = get_connection(db_path)
    total_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()['c']
    total_listings = conn.execute("SELECT COUNT(*) AS c FROM listings").fetchone()['c']
    total_messages = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()['c']
    exported = conn.execute("SELECT COUNT(*) AS c FROM listings WHERE exported_at IS NOT NULL").fetchone()['c']
    pending = total_listings - exported
    latest_listing = conn.execute(
        "SELECT created_at FROM listings ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return {
        'total_users': total_users,
        'total_listings': total_listings,
        'total_messages': total_messages,
        'exported_to_training': exported,
        'pending_export': pending,
        'latest_listing_at': latest_listing['created_at'] if latest_listing else None,
    }


def get_listings_timeline(db_path=None):
    """Every listing's creation timestamp, oldest first — used to chart how the
    live dataset has grown over time."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT created_at, brand, model, exported_at FROM listings ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
