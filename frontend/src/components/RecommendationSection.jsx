import RecommendationCard from "./RecommendationCard";

export default function RecommendationSection({
  title,
  subtitle,
  items,
  onOpen,
  onFeedback,
  busyItem,
  emptyText,
}) {
  return (
    <section className="mb-8">
      <div className="mb-4 flex items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-bold text-ink md:text-2xl">{title}</h2>
          <p className="text-sm text-slate-600">{subtitle}</p>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="glass rounded-2xl p-6 text-sm text-slate-500">{emptyText}</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item, index) => (
            <RecommendationCard
              key={`${title}-${item.id}-${index}`}
              item={item}
              busy={busyItem === item.title}
              delay={index * 0.05}
              onOpen={() => onOpen(item)}
              onFeedback={(eventType) => onFeedback(item, eventType)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
