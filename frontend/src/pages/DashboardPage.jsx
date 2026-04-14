import { useNavigate } from "react-router-dom";
import { useLearningStore } from "../store/useLearningStore";

export default function DashboardPage() {
  const navigate = useNavigate();
  const state = useLearningStore((store) => store);
  const prediction = state.prediction;
  const recommendations = state.recommendations;
  const recommendationCards = state.recommendationCards;
  const loading = state.loading;
  const error = state.error;
  const userInput = state.userInput;
  const analyzeUser = state.analyzeUser;

  console.log("State:", state);

  if (!prediction) {
    return (
      <section className="animate-rise">
        {error ? <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div> : null}
        <div className="glass rounded-2xl p-10 text-center shadow-card">
          <p className="font-display text-2xl font-bold text-ink">Loading...</p>
          <p className="mt-2 text-sm text-slate-600">
            {loading ? "Analyzing your profile and preparing the dashboard." : "Waiting for prediction data from the onboarding analysis."}
          </p>
        </div>
      </section>
    );
  }

  if (!recommendations.length) {
    return (
      <section className="animate-rise">
        {error ? <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div> : null}
        <div className="glass rounded-2xl p-10 text-center shadow-card">
          <p className="font-display text-2xl font-bold text-ink">Loading...</p>
          <p className="mt-2 text-sm text-slate-600">
            {loading ? "Preparing personalized recommendations." : "Waiting for recommendation data from the onboarding analysis."}
          </p>
        </div>
      </section>
    );
  }

  const predictedScore = Number(prediction?.predicted_score ?? 0);
  const riskLevel = prediction?.risk_level ?? "medium";

  return (
    <section className="animate-rise">
      {error ? <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div> : null}

      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <article className="glass rounded-3xl p-6 shadow-card">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Prediction Card</p>
          <div className="mt-4 flex items-start justify-between gap-4">
            <div>
              <p className="text-sm text-slate-600">Predicted score</p>
              <p className="mt-2 font-display text-4xl font-bold text-ink">
                {Number.isFinite(predictedScore) ? predictedScore.toFixed(1) : "0.0"}
              </p>
            </div>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-slate-700">
              {riskLevel} risk
            </span>
          </div>

          <p className="mt-4 text-sm text-slate-600">
            {riskLevel === "high"
              ? "The model sees elevated risk. Focus on the recommendations below."
              : riskLevel === "medium"
                ? "The model sees moderate risk. Consistent practice should improve the forecast."
                : "The model sees low risk. Keep the current momentum going."}
          </p>
        </article>

        <article className="glass rounded-3xl p-6 shadow-card">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Recommendations</p>
          <div className="mt-4 space-y-3">
            {recommendations.map((item, index) => {
              const card = recommendationCards[index];

              return (
                <button
                  key={`${item}-${index}`}
                  type="button"
                  onClick={() => {
                    if (card) {
                      navigate(`/learn/${card.id}`);
                    }
                  }}
                  className="w-full rounded-2xl border border-slate-200 bg-white/80 p-4 text-left text-sm text-slate-700 transition hover:border-slate-300 hover:bg-white"
                >
                  {item}
                </button>
              );
            })}
          </div>
        </article>
      </div>

      <div className="mt-6 rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">
        <button
          type="button"
          onClick={async () => {
            if (!userInput) {
              return;
            }

            try {
              await analyzeUser(userInput);
            } catch (err) {
              console.error("API Error:", err);
            }
          }}
          className="font-semibold text-ink underline decoration-slate-400 underline-offset-4"
        >
          Refresh analysis
        </button>
      </div>
    </section>
  );
}

