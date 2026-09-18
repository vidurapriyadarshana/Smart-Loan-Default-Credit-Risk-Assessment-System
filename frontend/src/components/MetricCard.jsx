import React from 'react';

export default function MetricCard({ label, value, subtext, icon: Icon, color = "text-sky-400" }) {
  return (
    <div className="glass-panel p-4 rounded-xl border border-white/5 flex items-start gap-3.5 transition-all duration-200 hover:border-white/15">
      {Icon && (
        <div className={`p-2.5 rounded-lg bg-white/5 ${color}`}>
          <Icon className="w-5 h-5" />
        </div>
      )}
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</p>
        <p className="text-xl font-bold text-white font-heading mt-0.5">{value}</p>
        {subtext && <p className="text-[11px] text-slate-500 mt-0.5">{subtext}</p>}
      </div>
    </div>
  );
}
