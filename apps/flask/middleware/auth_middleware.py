from functools import wraps
from flask import request, jsonify
from pymongo import MongoClient
from datetime import datetime, timezone
import urllib.parse
import os
import logging

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. PRIORITAS: Ambil token dari header Authorization
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token_from_header = auth_header.split(' ')[1]
            # URL decode jika ada encoding
            token_from_header = urllib.parse.unquote(token_from_header)
            # Jika token mengandung titik (signature), ambil bagian pertama
            session_token = urllib.parse.unquote(token_from_header)
            logger.debug(f"Validating token from Authorization header: {session_token[:10]}...")
            
            now_utc = datetime.now(timezone.utc)
            session = db.session.find_one({
                "token": session_token,
                "expiresAt": {"$gt": now_utc}
            })
            if session:
                request.session = session
                return f(*args, **kwargs)
            else:
                logger.warning(f"Invalid token from Authorization header: {session_token[:10]}...")
                # Jangan langsung gagal, fallback ke cookie (untuk kompatibilitas localhost)

        # 2. FALLBACK: Ambil dari cookie (untuk localhost atau jika header tidak ada)
        raw_token = (request.cookies.get('__Secure-better-auth.session_token') or
                     request.cookies.get('better-auth.session_token'))
        
        if not raw_token:
            logger.warning(f"Unauthorized: No session token in Authorization header or cookie. Path={request.path}. Cookies received: {list(request.cookies.keys())}")
            return jsonify({
                "success": False,
                "error": {"code": "UNAUTHORIZED", "message": "Session required. Please login."}
            }), 401

        decoded_token = urllib.parse.unquote(raw_token)
        session_token = decoded_token.split('.')[0] if '.' in decoded_token else decoded_token
        logger.debug(f"Validating token from cookie: {session_token[:10]}...")
        
        now_utc = datetime.now(timezone.utc)
        session = db.session.find_one({
            "token": session_token,
            "expiresAt": {"$gt": now_utc}
        })
        
        if not session:
            logger.warning(f"Invalid or expired session from cookie: {session_token[:10]}...")
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Invalid or expired session. Please login."
                }
            }), 401
        
        request.session = session
        return f(*args, **kwargs)
    return decorated_function