import { motion } from "framer-motion";
import FeedbackButtons from "./FeedbackButtons";

function difficultyClass(level) {
  if (level === "Starter") {
    return "bg-emerald-100 text-emerald-700";
  }
  if (level === "Challenge") {
    return "bg-amber-100 text-amber-700";
  }
  return "bg-sky-100 text-sky-700";
}

function riskClass(riskLevel) {
  if (riskLevel === "high") {
    return "bg-rose-100 text-rose-700 border-rose-200";
  }
  if (riskLevel === "medium") {
    return "bg-amber-100 text-amber-700 border-amber-200";
  }
  return "bg-emerald-100 text-emerald-700 border-emerald-200";
}

export default function RecommendationCard({ item, onFeedback, onOpen, busy = false, delay = 0 }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35 }}
      className="glass flex h-full flex-col rounded-2xl p-4 shadow-card"
    >
      <div className="mb-4 flex items-start justify-between gap-2">
        <h3 className="font-display text-lg font-semibold text-ink">{item.title}</h3>
        <div className="flex flex-wrap justify-end gap-2">
          <span className={`rounded-full px-2 py-1 text-xs font-semibold ${difficultyClass(item.difficulty)}`}>{item.difficulty}</span>
          <span className={`rounded-full border px-2 py-1 text-xs font-semibold capitalize ${riskClass(item.riskLevel)}`}>{item.riskLevel} risk</span>
        </div>
      </div>

      <div className="mb-3 flex items-center gap-2">
        <span className="rounded-lg bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700">Predicted score: {item.predictedScore}</span>
      </div>

      <p className="mb-2 text-sm text-slate-600">{item.reason}</p>
      <p className="mb-5 text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{item.estimatedMinutes} min focus sprint</p>

      <div className="mt-auto space-y-3">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onOpen}
            className="flex-1 rounded-xl bg-ink px-3 py-2 text-center text-sm font-semibold text-white transition hover:bg-slate-800"
          >
            Start learning
          </button>
        </div>

        <FeedbackButtons disabled={busy} compact onAction={onFeedback} />
      </div>
    </motion.article>
  );
}
