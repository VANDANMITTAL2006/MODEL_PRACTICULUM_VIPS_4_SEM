import { useNavigate } from "react-router-dom";
import PerformanceCard from "../components/PerformanceCard";
import RecommendationSection from "../components/RecommendationSection";
import { useLearningStore } from "../store/useLearningStore";

function RiskPanel({ topics }) {
  return (
    <article className="glass rounded-2xl p-5 shadow-card">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Risk Panel</p>
      <div className="mt-3 grid gap-3">
        <div className="rounded-xl bg-rose-50 p-3 text-rose-700">
          <p className="text-xs font-semibold uppercase tracking-[0.14em]">High Risk Topics</p>
          <p className="mt-1 text-sm">{topics.high.length ? topics.high.join(" | ") : "No high-risk topics right now."}</p>
        </div>
        <div className="rounded-xl bg-amber-50 p-3 text-amber-700">
          <p className="text-xs font-semibold uppercase tracking-[0.14em]">Medium Risk Topics</p>
          <p className="mt-1 text-sm">{topics.medium.length ? topics.medium.join(" | ") : "No medium-risk topics right now."}</p>
        </div>
        <div className="rounded-xl bg-emerald-50 p-3 text-emerald-700">
          <p className="text-xs font-semibold uppercase tracking-[0.14em]">Low Risk Topics</p>
          <p className="mt-1 text-sm">{topics.low.length ? topics.low.join(" | ") : "No low-risk topics right now."}</p>
        </div>
      </div>
    </article>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const prediction = useLearningStore((state) => state.prediction);
  const loading = useLearningStore((state) => state.loading);
  const error = useLearningStore((state) => state.error);
  const recommendationSections = useLearningStore((state) => state.recommendationSections);
  const riskTopics = useLearningStore((state) => state.riskTopics);
  const userInput = useLearningStore((state) => state.userInput);
  const analyzeUser = useLearningStore((state) => state.analyzeUser);
  const setTopicFeedback = useLearningStore((state) => state.setTopicFeedback);
  const startLearningTopic = useLearningStore((state) => state.startLearningTopic);
  const getInsightText = useLearningStore((state) => state.getInsightText);

  const hasData = Boolean(prediction && recommendationSections.recommended.length > 0);

  const openLearning = (item) => {
    startLearningTopic(item.id);
    navigate(`/learn/${item.id}`);
  };

  const handleFeedback = async (item, feedbackType) => {
    setTopicFeedback(item.id, feedbackType);
    if (!userInput) {
      return;
    }

    try {
      const bump = feedbackType === "like" ? 1.5 : feedbackType === "dislike" ? -1.5 : 0;
      await analyzeUser({
        ...userInput,
        engagement_score: Math.max(0, Math.min(100, Number(userInput.engagement_score || 50) + bump)),
      });
    } catch (err) {
      console.error("feedback re-analysis failed", err);
    }
  };

  if (loading && !hasData) {
    return (
      <div className="glass animate-rise rounded-2xl p-10 text-center shadow-card">
        <p className="font-display text-2xl font-bold text-ink">Analyzing your profile...</p>
        <p className="mt-2 text-sm text-slate-600">We are generating a personalized learning forecast and topic feed.</p>
      </div>
    );
  }

  return (
    <section className="animate-rise">
      <PerformanceCard
        prediction={prediction}
        insightText={getInsightText()}
        refreshing={loading}
        onRefresh={() => {
          if (!userInput) {
            return;
          }
          analyzeUser(userInput).catch((err) => {
            console.error("manual dashboard refresh failed", err);
          });
        }}
      />

      <div className="mb-6 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <article className="glass rounded-2xl p-5 shadow-card">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">AI Insight</p>
          <p className="mt-2 text-sm text-slate-700">{getInsightText()}</p>
        </article>
        <RiskPanel topics={riskTopics} />
      </div>

      {error ? <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div> : null}

      {!hasData && !loading ? (
        <div className="glass rounded-2xl p-6 text-sm text-slate-600">No recommendations yet. Return to onboarding and run analysis.</div>
      ) : (
        <>
          <RecommendationSection
            title="Recommended For You"
            subtitle="Generated from your latest prediction and risk profile"
            items={recommendationSections.recommended}
            onOpen={openLearning}
            onFeedback={handleFeedback}
            emptyText="No recommendations available yet."
          />

          <RecommendationSection
            title="Continue Learning"
            subtitle="Quick wins to keep your streak alive"
            items={recommendationSections.continueLearning}
            onOpen={openLearning}
            onFeedback={handleFeedback}
            emptyText="Start a topic to populate this section."
          />
        </>
      )}
    </section>
  );
}
