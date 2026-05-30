const API_PORT = "8000";
const FALLBACK_API_BASE = "http://192.168.137.47:8000";

const getApiBase = () => {
  const envApiBase = import.meta.env.VITE_API_BASE;
  if (envApiBase) {
    return envApiBase;
  }

  if (typeof window === "undefined" || !window.location.hostname) {
    return FALLBACK_API_BASE;
  }

  return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
};

export const API_BASE = getApiBase();
