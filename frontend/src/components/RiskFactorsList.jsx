import React from 'react';
import { Info, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function RiskFactorsList({ factors, isDeclined }) {
  if (!factors || factors.length === 0) return null;

  return (
    <div className="glass-panel p-5 rounded-2xl border border-white/5 mt-5">
      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
        <Info className="w-4 h-4 text-sky-400" />
        Key Decision & Risk Explanations
      </h4>
      <div className="space-y-2.5">
        {factors.map((factor, idx) => {
          const isWarning = factor.includes('Critical') || factor.includes('Subprime') || factor.includes('default');
          return (
            <div
              key={idx}
              className={`p-3 rounded-lg text-xs leading-relaxed flex items-start gap-2.5 border ${
                isWarning
                  ? 'bg-rose-500/10 text-rose-300 border-rose-500/20'
                  : 'bg-slate-800/40 text-slate-300 border-white/5'
              }`}
            >
              {isWarning ? (
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
              )}
              <span>{factor}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
