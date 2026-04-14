import axios from "axios";

const apiClient = axios.create({
  baseURL: "https://model-practiculum-vips-4-sem.onrender.com",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error?.response?.data?.detail ||
      error?.response?.data?.message ||
      error?.message ||
      "Request failed";

    return Promise.reject({
      status: error?.response?.status,
      detail,
      raw: error,
    });
  }
);

export default apiClient;
