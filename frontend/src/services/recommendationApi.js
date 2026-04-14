import apiClient from "./apiClient";

export async function analyzeUser(data = {}) {
  try {
    const response = await apiClient.post("/analyze-user", data);
    const responseData = response?.data;

    console.log("API response:", responseData);

    if (!responseData || typeof responseData !== "object") {
      throw new Error("Invalid API response: response data is missing.");
    }

    if (!responseData.prediction) {
      throw new Error("Invalid API response: prediction is missing.");
    }

    if (!Array.isArray(responseData.recommendations)) {
      throw new Error("Invalid API response: recommendations must be an array.");
    }

    return responseData;
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}
