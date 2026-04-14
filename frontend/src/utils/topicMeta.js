const DIFFICULTY_BANDS = [
  { key: "Starter", min: 0, max: 45 },
  { key: "Guided", min: 46, max: 70 },
  { key: "Challenge", min: 71, max: 100 },
];

export function getDifficultyLabel(quizScore = 60, offset = 0) {
  const adjusted = Math.max(0, Math.min(100, Number(quizScore) + offset));
  const band = DIFFICULTY_BANDS.find((item) => adjusted >= item.min && adjusted <= item.max);
  return band?.key || "Guided";
}

export function whyRecommended(topic, user) {
  const reasons = [
    `Because you struggled with ${user.subjectWeakness}`,
    "Based on your recent activity",
    `Matched to your ${user.skillLevel.toLowerCase()} learning pace`,
  ];

  const hash = topic.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return reasons[hash % reasons.length];
}

export function riskFromScore(score) {
  if (score < 50) return "high";
  if (score < 70) return "medium";
  return "low";
}

export function toRecommendationCard(topicData, index, user) {
  const topic = typeof topicData === "string" ? topicData : topicData.topic;
  const predictedScore =
    typeof topicData === "object" && topicData !== null && typeof topicData.predicted_score === "number"
      ? topicData.predicted_score
      : Math.max(0, Math.min(100, Number(user.quizScore) - 4 + index * 1.5));
  const riskLevel =
    typeof topicData === "object" && topicData !== null && topicData.risk_level
      ? topicData.risk_level
      : riskFromScore(predictedScore);
  const reason =
    typeof topicData === "object" && topicData !== null && topicData.reason
      ? topicData.reason
      : whyRecommended(topic, user);

  return {
    id: topic.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    title: topic,
    difficulty: getDifficultyLabel(user.quizScore, index * 2 - 3),
    reason,
    predictedScore: Number(predictedScore.toFixed(1)),
    riskLevel,
    estimatedMinutes: 8 + ((index % 4) + 1) * 4,
  };
}
