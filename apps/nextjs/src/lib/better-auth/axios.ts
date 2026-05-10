import axios from "axios";
export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_FLASK_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Interceptor request: tambahkan Authorization header jika ada token
api.interceptors.request.use((config) => {
  // Header untuk ngrok
  config.headers["ngrok-skip-browser-warning"] = "69420";

  const token = getSessionToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

function getCookie(name: string): string | null {
  if (typeof document === "undefined") return null;

  const match = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));

  return match ? decodeURIComponent(match[2]) : null;
}

function getSessionToken(): string | null {
  return (
    getCookie("__Secure-better-auth.session_token") ||
    getCookie("better-auth.session_token")
  );
}
