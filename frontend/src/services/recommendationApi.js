import apiClient from "./apiClient";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value)));
}

export function toAnalyzePayload(input = {}) {
  const quizScore = clamp(input.quiz_score ?? input.quizScore ?? 65, 0, 100);
  const timeSpentHours = clamp(input.time_spent_hours ?? input.timeSpentHours ?? 4, 0.1, 24);
  const attendance = clamp(input.attendance ?? 70, 0, 100);
  const engagement = clamp(input.engagement_score ?? input.engagementScore ?? 50, 0, 100);
  const consistency = clamp(input.consistency_score ?? input.consistencyScore ?? 60, 0, 100);
  const previousScore = clamp(input.previous_score ?? input.previousScore ?? quizScore, 0, 100);
  const subjectWeakness = String(input.subject_weakness ?? input.subjectWeakness ?? "Algebra").trim() || "Algebra";

  return {
    quiz_score: quizScore,
    time_spent_hours: timeSpentHours,
    attendance,
    engagement_score: engagement,
    consistency_score: consistency,
    previous_score: previousScore,
    subject_weakness: subjectWeakness,
  };
}

export async function analyzeUser(payload) {
  const requestBody = toAnalyzePayload(payload);
  const { data } = await apiClient.post("/analyze-user", requestBody);
  return data;
}
