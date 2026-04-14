import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLearningStore } from "../store/useLearningStore";

const SUBJECTS = ["Algebra", "Geometry", "Statistics", "Physics", "Chemistry", "Computer Science", "Calculus"];

export default function OnboardingPage() {
  const navigate = useNavigate();
  const analyzeUser = useLearningStore((state) => state.analyzeUser);
  const error = useLearningStore((state) => state.error);
  const loading = useLearningStore((state) => state.loading);

  const [form, setForm] = useState({
    quiz_score: 62,
    time_spent_hours: 4.0,
    attendance: 80,
    engagement_score: 50,
    consistency_score: 60,
    previous_score: 58,
    subject_weakness: "Algebra",
  });

  const updateField = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({
      ...current,
      [name]:
        [
          "quiz_score",
          "time_spent_hours",
          "attendance",
          "engagement_score",
          "consistency_score",
          "previous_score",
        ].includes(name)
          ? Number(value)
          : value,
    }));
  };

  const submit = async (event) => {
    event.preventDefault();
    try {
      await analyzeUser(form);
      navigate("/dashboard");
    } catch (err) {
      console.error("onboarding analyze failed", err);
    }
  };

  return (
    <section className="mx-auto max-w-4xl animate-rise">
      <div className="glass overflow-hidden rounded-3xl shadow-card">
        <div className="grid md:grid-cols-2">
          <div className="bg-ink p-8 text-white">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.24em] text-slate-300">Onboarding</p>
            <h2 className="font-display text-3xl font-bold">Train your AI learning profile.</h2>
            <p className="mt-4 text-sm text-slate-300">
              Enter your current learning signals. We run AI analysis first, then build a personalized feed before you reach dashboard.
            </p>
          </div>

          <form className="grid gap-4 p-8" onSubmit={submit}>
            <label className="text-sm font-semibold text-slate-700">
              Current weak area
              <select
                name="subject_weakness"
                value={form.subject_weakness}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              >
                {SUBJECTS.map((subject) => (
                  <option key={subject}>{subject}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Quiz score
              <input
                name="quiz_score"
                type="number"
                min="0"
                max="100"
                value={form.quiz_score}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Time spent (hours/day)
              <input
                name="time_spent_hours"
                type="number"
                min="0.1"
                max="24"
                step="0.1"
                value={form.time_spent_hours}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Attendance
              <input
                name="attendance"
                type="number"
                min="0"
                max="100"
                value={form.attendance}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Engagement (0-100)
              <input
                name="engagement_score"
                type="number"
                min="0"
                max="100"
                value={form.engagement_score}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Consistency (0-100)
              <input
                name="consistency_score"
                type="number"
                min="0"
                max="100"
                value={form.consistency_score}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <label className="text-sm font-semibold text-slate-700">
              Previous score
              <input
                name="previous_score"
                type="number"
                min="0"
                max="100"
                value={form.previous_score}
                onChange={updateField}
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 outline-none ring-sky-200 focus:ring"
              />
            </label>

            <button
              type="submit"
              disabled={loading}
              className="mt-2 rounded-xl bg-sunrise px-4 py-2 font-semibold text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? "Analyzing your learning profile..." : "Analyze My Learning"}
            </button>

            {error ? <p className="text-sm text-rose-600">{error}</p> : null}
          </form>
        </div>
      </div>
    </section>
  );
}
