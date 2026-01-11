import React, { useState, useEffect } from 'react';
import { Play, Pause, ChevronLeft, ChevronRight, User, TrendingUp, Zap, Target, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import sequencesData from '../data/sequences.json';

const Analysis = () => {
    const [selectedSeq, setSelectedSeq] = useState(sequencesData[0]);
    const [isPlaying, setIsPlaying] = useState(false);

    // Get unique teams
    const teams = [...new Set(sequencesData.map(s => s.team_name))];
    const [selectedTeam, setSelectedTeam] = useState(teams[0]);

    // Filter sequences by team
    const filteredSequences = sequencesData.filter(s => s.team_name === selectedTeam);

    // Update selected sequence when team changes
    useEffect(() => {
        if (filteredSequences.length > 0 && !filteredSequences.find(s => s.id === selectedSeq.id)) {
            setSelectedSeq(filteredSequences[0]);
        }
    }, [selectedTeam]);

    const handleSeqChange = (id) => {
        const seq = sequencesData.find(s => s.id === id);
        if (seq) {
            setSelectedSeq(seq);
            setIsPlaying(false);
        }
    };

    // Extract chart data from actions
    const chartData = selectedSeq.actions.map((action, index) => ({
        name: `Act ${index + 1}`,
        vaep: action.vaep,
        type: action.type
    }));

    return (
        <div className="pt-[72px] flex h-screen overflow-hidden">
            {/* List Sidebar */}
            <aside className="w-[300px] border-r border-white/5 bg-slate-900/20 flex flex-col transition-all">
                <div className="p-6 border-b border-white/5">
                    <h2 className="text-sm font-bold mb-4 flex items-center gap-2 uppercase tracking-wider text-slate-400">
                        <Target size={16} />
                        Filter Team
                    </h2>
                    <div className="relative">
                        <select
                            className="w-full bg-slate-800 border border-white/10 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-indigo-500 appearance-none font-medium cursor-pointer"
                            onChange={(e) => setSelectedTeam(e.target.value)}
                            value={selectedTeam}
                        >
                            {teams.map(team => (
                                <option key={team} value={team}>{team}</option>
                            ))}
                        </select>
                        <ChevronRight className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 rotate-90 pointer-events-none" size={16} />
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    <div className="px-2 pb-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
                        Sequences ({filteredSequences.length})
                    </div>
                    {filteredSequences.map(s => (
                        <div
                            key={s.id}
                            onClick={() => handleSeqChange(s.id)}
                            className={`p-4 rounded-xl cursor-pointer transition-all border ${selectedSeq.id === s.id
                                ? 'bg-indigo-600/20 border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.2)]'
                                : 'bg-white/5 border-transparent hover:bg-white/10'}`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <div className="flex flex-col gap-1">
                                    <span className={`text-xs font-bold uppercase tracking-wider ${selectedSeq.id === s.id ? 'text-indigo-300' : 'text-slate-500'}`}>#{s.id}</span>
                                    {s.cluster !== -1 && (
                                        <span className="text-[9px] font-bold text-cyan-500 uppercase tracking-tighter">Cluster {s.cluster}</span>
                                    )}
                                </div>
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${s.result === 'success' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-500/20 text-slate-400'}`}>
                                    {s.result}
                                </span>
                            </div>
                            <div className="text-[11px] text-slate-500 flex gap-3">
                                <span>{s.duration.toFixed(1)}s</span>
                                <span>{s.actions.length} acts</span>
                                <span className="text-indigo-400/80">{s.outcome}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </aside>

            {/* Main Analysis Area */}
            <main className="flex-1 overflow-y-auto bg-[#020617] p-10">
                <div className="max-w-[1600px] mx-auto space-y-12">
                    {/* Video Section */}
                    <div className="glass-card overflow-hidden p-0 relative group">
                        <div className="aspect-video bg-black flex items-center justify-center relative">
                            <video
                                key={selectedSeq.id}
                                src={selectedSeq.video_url}
                                className="w-full h-full object-contain"
                                onPlay={() => setIsPlaying(true)}
                                onPause={() => setIsPlaying(false)}
                                controls
                            />
                            {!isPlaying && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black/40 pointer-events-none group-hover:bg-black/20 transition-all">
                                    <div className="w-20 h-20 bg-indigo-600 rounded-full flex items-center justify-center shadow-2xl scale-110">
                                        <Play className="text-white fill-current translate-x-1" size={32} />
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="p-4 bg-slate-900/80 border-t border-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 bg-indigo-600/20 rounded-lg flex items-center justify-center">
                                    <Play size={20} className="text-indigo-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold">{selectedSeq.team_name} - Tactical Animation</h3>
                                    <p className="text-xs text-slate-500">Visualization of offensive sequence pattern</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* VAEP Timeline Chart */}
                    <div className="glass-card">
                        <div className="flex items-center justify-between mb-10">
                            <h3 className="text-xl font-bold flex items-center gap-2">
                                <TrendingUp className="text-emerald-400" size={20} />
                                VAEP Contribution Timeline
                            </h3>
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                                <div className="w-3 h-3 bg-indigo-500 rounded-full" />
                                <span>Value Added Per Action</span>
                            </div>
                        </div>

                        <div className="h-[280px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="colorVaep" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                                    <XAxis
                                        dataKey="name"
                                        stroke="#475569"
                                        fontSize={11}
                                        tickLine={false}
                                        axisLine={false}
                                        dy={10}
                                    />
                                    <YAxis
                                        stroke="#475569"
                                        fontSize={11}
                                        tickLine={false}
                                        axisLine={false}
                                        tickFormatter={(val) => val.toFixed(2)}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                        itemStyle={{ color: '#818cf8', fontWeight: 'bold' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="vaep"
                                        stroke="#6366f1"
                                        strokeWidth={3}
                                        fillOpacity={1}
                                        fill="url(#colorVaep)"
                                        animationDuration={1500}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </main>

            {/* Stats Sidebar */}
            <aside className="w-[300px] border-l border-white/5 bg-slate-900/40 p-6 space-y-12 overflow-y-auto transition-all">
                <section className="mb-12">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-8">Sequence Stats</h4>
                    <div className="space-y-6">
                        <StatRow label="Outcome" value={selectedSeq.outcome} icon={<Target size={14} />} />
                        <StatRow label="Duration" value={`${selectedSeq.duration.toFixed(1)}s`} icon={<Zap size={14} />} />
                        <StatRow label="Avg Speed" value={`${selectedSeq.speed.toFixed(1)} m/s`} icon={<TrendingUp size={14} />} />
                        <StatRow label="Distance" value={`${selectedSeq.distance.toFixed(1)}m`} icon={<TrendingUp size={14} />} />
                    </div>
                </section>

                <section className="bg-indigo-600/10 rounded-[20px] p-6 border border-indigo-500/20 mb-12">
                    <div className="flex items-center gap-2 mb-4 text-indigo-400">
                        <User size={16} />
                        <h4 className="text-xs font-bold uppercase tracking-wider">Key Player (MVP)</h4>
                    </div>
                    <div className="text-2xl font-bold mb-3 font-display">{selectedSeq.mvp}</div>
                    <p className="text-xs text-indigo-400/80 leading-relaxed font-light">
                        해당 전술 시퀀스에서 가장 높은 VAEP 기여도를 보여주었습니다. 결정적인 기점을 만들거나 기회를 창출했습니다.
                    </p>
                </section>

                <section>
                    <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-8 font-display">Action History</h4>
                    <div className="space-y-4">
                        {selectedSeq.actions.slice(-5).map((action, i) => (
                            <div key={i} className="flex items-start gap-4">
                                <div className="w-2 h-2 rounded-full bg-indigo-500 mt-2 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                                <div>
                                    <div className="text-sm font-semibold">{action.type}</div>
                                    <div className="text-[11px] text-slate-500">{action.player}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            </aside>
        </div>
    );
};

const StatRow = ({ label, value, icon }) => (
    <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-400 text-sm">
            {icon}
            <span>{label}</span>
        </div>
        <div className="font-bold text-white capitalize">{value}</div>
    </div>
);

export default Analysis;
