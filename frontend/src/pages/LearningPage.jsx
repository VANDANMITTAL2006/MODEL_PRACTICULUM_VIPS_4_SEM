import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import FeedbackButtons from "../components/FeedbackButtons";
import LearningTracker from "../components/LearningTracker";
import { useLearningStore } from "../store/useLearningStore";

export default function LearningPage() {
  const navigate = useNavigate();
  const { itemId } = useParams();

  const recommendations = useLearningStore((state) => state.recommendations);
  const learningStats = useLearningStore((state) => state.learningStats);
  const startLearningTopic = useLearningStore((state) => state.startLearningTopic);
  const updateLearningTimer = useLearningStore((state) => state.updateLearningTimer);
  const setTopicFeedback = useLearningStore((state) => state.setTopicFeedback);
  const completeLearningTopic = useLearningStore((state) => state.completeLearningTopic);

  const currentItem = useMemo(() => recommendations.find((entry) => entry.id === itemId) || null, [itemId, recommendations]);

  if (!currentItem) {
    return (
      <div className="glass rounded-2xl p-6 text-sm text-slate-600">
        Topic not found. Go back to dashboard and start from a recommendation card.
      </div>
    );
  }

  const currentStats = learningStats[currentItem.id] || {};

  return (
    <section className="grid animate-rise gap-6 lg:grid-cols-[2fr_1fr]">
      <article className="glass rounded-3xl p-6 shadow-card">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Learning Module</p>
        <h2 className="font-display text-3xl font-bold text-ink">{currentItem.title}</h2>
        <p className="mt-3 text-sm text-slate-600">{currentItem.reason}</p>

        <div className="my-6 rounded-2xl border border-dashed border-slate-300 bg-white/70 p-5">
          <h3 className="font-display text-xl font-semibold text-ink">Focused Learning Sprint</h3>
          <p className="mt-2 text-sm leading-7 text-slate-700">
            Read the concept, simulate practice, and complete this sprint. Your interaction data updates the prediction model and
            refreshes recommendations on completion.
          </p>
        </div>

        <div className="rounded-2xl bg-white/80 p-4">
          <p className="mb-2 text-sm font-semibold text-slate-700">Was this recommendation useful?</p>
          <FeedbackButtons
            onAction={(feedbackType) => {
              setTopicFeedback(currentItem.id, feedbackType);
            }}
          />
        </div>

        <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
          <span>Status: {currentStats.completed ? "Completed" : "In progress"}</span>
          <button
            type="button"
            onClick={() => navigate("/dashboard")}
            className="rounded-lg border border-slate-300 px-3 py-1.5 font-semibold hover:bg-white"
          >
            Back to dashboard
          </button>
        </div>
      </article>

      <LearningTracker
        itemTitle={currentItem.title}
        onTick={(elapsedSeconds) => {
          if (!currentStats.clicked) {
            startLearningTopic(currentItem.id);
          }
          updateLearningTimer(currentItem.id, elapsedSeconds);
        }}
        onComplete={async (elapsedSeconds) => {
          updateLearningTimer(currentItem.id, elapsedSeconds);
          await completeLearningTopic(currentItem.id);
          navigate("/dashboard");
        }}
      />
    </section>
  );
}
