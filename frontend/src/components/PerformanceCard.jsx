function riskTone(riskLevel) {
  if (riskLevel === "high") {
    return "text-rose-700 bg-rose-100";
  }
  if (riskLevel === "medium") {
    return "text-amber-700 bg-amber-100";
  }
  return "text-emerald-700 bg-emerald-100";
}

export default function PerformanceCard({ prediction, insightText, refreshing, onRefresh }) {
  const safePrediction = prediction || {
    predictedScore: 0,
    riskLevel: "medium",
    confidence: 0,
    explanation: "We are collecting more interaction signals to improve prediction quality.",
  };

  const normalizedPrediction = {
    predictedScore: Number(safePrediction.predictedScore ?? safePrediction.predicted_score ?? 0),
    riskLevel: safePrediction.riskLevel ?? safePrediction.risk_level ?? "medium",
  };

  return (
    <section className="mb-8 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
      <article className="glass rounded-2xl p-5 shadow-card">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Predicted Performance</p>
            <p className="mt-2 font-display text-4xl font-bold text-ink">{normalizedPrediction.predictedScore.toFixed(1)}</p>
          </div>
          <span className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${riskTone(normalizedPrediction.riskLevel)}`}>
            {normalizedPrediction.riskLevel} risk
          </span>
        </div>

        <p className="mt-3 text-sm text-slate-600">{insightText || safePrediction.explanation || "Your model forecast updates as you learn."}</p>

        <div className="mt-4 flex items-center justify-between gap-3">
          <p className="text-sm font-semibold text-slate-700">Risk-adjusted AI forecast</p>
          <button
            type="button"
            onClick={onRefresh}
            className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800"
          >
            {refreshing ? "Refreshing..." : "Refresh insight"}
          </button>
        </div>
      </article>
    </section>
  );
}
