import { create } from "zustand";
import { persist } from "zustand/middleware";
import { analyzeUser as analyzeUserApi } from "../services/recommendationApi";

function buildInsight(prediction) {
  if (!prediction) {
    return "Complete onboarding to see personalized learning insights.";
  }

  const score = Number(prediction.predicted_score ?? 0).toFixed(1);
  if (prediction.risk_level === "high") {
    return `You are currently at risk. Focus on weak concepts and daily practice to lift your forecasted score of ${score}.`;
  }

  if (prediction.risk_level === "medium") {
    return `You are on track with room to improve. Consistent effort can push your forecasted score above ${score}.`;
  }

  return `Strong momentum detected. Keep reinforcing your strengths to maintain a forecasted score around ${score}.`;
}

function slugify(value, index) {
  const slug = String(value || "recommendation")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return `${slug || "recommendation"}-${index + 1}`;
}

function buildRecommendationCards(recommendations = [], prediction) {
  return recommendations.map((recommendation, index) => ({
    id: slugify(recommendation, index),
    title: String(recommendation),
    reason:
      prediction?.risk_level === "high"
        ? "Priority topic to reduce near-term learning risk."
        : prediction?.risk_level === "medium"
          ? "Recommended to improve consistency in your weak areas."
          : "Great next step to preserve your current momentum.",
    predictedScore: Number(prediction?.predicted_score ?? 0),
    riskLevel: prediction?.risk_level ?? "medium",
    estimatedMinutes: 10 + (index % 4) * 5,
  }));
}

export const useLearningStore = create(
  persist(
    (set, get) => ({
      userInput: null,
      prediction: null,
      recommendations: [],
      recommendationCards: [],
      loading: false,
      error: "",
      activeTopicId: null,
      learningStats: {},

      setPrediction: (prediction) => set({ prediction }),

      setRecommendations: (recommendations) =>
        set((state) => ({
          recommendations,
          recommendationCards: buildRecommendationCards(recommendations, state.prediction),
        })),

      analyzeUser: async (input) => {
        set({ loading: true, error: "", userInput: input });

        try {
          const response = await analyzeUserApi(input);
          const prediction = response?.prediction || null;
          const recommendations = Array.isArray(response?.recommendations) ? response.recommendations.filter(Boolean).map(String) : [];

          set({
            prediction,
            recommendations,
            recommendationCards: buildRecommendationCards(recommendations, prediction),
            loading: false,
            error: "",
          });

          return response;
        } catch (error) {
          console.error("analyzeUser action failed", error);
          set({ loading: false, error: error?.detail || error?.message || "Unable to analyze user right now." });
          throw error;
        }
      },

      startLearningTopic: (topicId) => {
        set((state) => ({
          activeTopicId: topicId,
          learningStats: {
            ...state.learningStats,
            [topicId]: {
              ...(state.learningStats[topicId] || {}),
              clicked: true,
              startedAt: Date.now(),
              elapsedSeconds: 0,
              completed: false,
              feedback: state.learningStats[topicId]?.feedback || null,
            },
          },
        }));
      },

      updateLearningTimer: (topicId, elapsedSeconds) => {
        set((state) => ({
          learningStats: {
            ...state.learningStats,
            [topicId]: {
              ...(state.learningStats[topicId] || {}),
              elapsedSeconds,
            },
          },
        }));
      },

      setTopicFeedback: (topicId, feedback) => {
        set((state) => ({
          learningStats: {
            ...state.learningStats,
            [topicId]: {
              ...(state.learningStats[topicId] || {}),
              feedback,
            },
          },
        }));
      },

      completeLearningTopic: async (topicId) => {
        set((state) => ({
          learningStats: {
            ...state.learningStats,
            [topicId]: {
              ...(state.learningStats[topicId] || {}),
              completed: true,
            },
          },
        }));

        const userInput = get().userInput;
        if (!userInput) {
          return;
        }

        const stats = get().learningStats[topicId] || {};
        const tunedPayload = {
          ...userInput,
          engagement_score: Math.min(100, Number(userInput.engagement_score || 50) + 2),
          consistency_score: Math.min(100, Number(userInput.consistency_score || 60) + (stats.completed ? 2 : 0)),
          previous_score: Math.max(0, Math.min(100, Number(userInput.previous_score || 60) + (stats.completed ? 1.5 : 0))),
          time_spent_hours: Math.max(
            0.1,
            Math.min(24, Number(userInput.time_spent_hours || 1) + Number(stats.elapsedSeconds || 0) / 3600)
          ),
        };

        try {
          await get().analyzeUser(tunedPayload);
        } catch (error) {
          console.error("re-analysis after learning failed", error);
        }
      },

      getInsightText: () => buildInsight(get().prediction),
    }),
    {
      name: "ai-learning-app-v2",
      partialize: (state) => ({
        userInput: state.userInput,
        prediction: state.prediction,
        recommendations: state.recommendations,
        recommendationCards: state.recommendationCards,
        activeTopicId: state.activeTopicId,
        learningStats: state.learningStats,
      }),
    }
  )
);

