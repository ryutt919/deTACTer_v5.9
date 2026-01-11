import React from 'react';
import { Play, TrendingUp, Cpu, Target, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

const Home = ({ onAnalyze }) => {
    return (
        <div className="pt-[72px]">
            {/* Hero Section */}
            <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
                {/* Animated Background Gradients - Adjusted Size & Position */}
                <div className="absolute top-[5%] left-[-20%] w-[800px] h-[400px] bg-indigo-600/20 blur-[100px] rounded-full animate-pulse" />
                <div className="absolute bottom-[-5%] right-[-20%] w-[800px] h-[400px] bg-cyan-600/20 blur-[100px] rounded-full animate-pulse" style={{ animationDelay: '2s' }} />

                <div className="container relative z-10 text-center">
                    <motion.div
                        className="-mt-32"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="text-7xl md:text-9xl font-bold mb-12 tracking-tighter">
                            <span className="text-white">deTACTer</span><br />
                        </h1>
                        <div className="container gap-5">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                                <FeatureCard
                                    icon={<TrendingUp className="text-indigo-4300" />}
                                    title="VAEP Quantification"
                                    description="모든 액션이 득점 확률에 미치는 실질적인 가치를 수치로 증명하여 전술의 효율성을 객관적으로 평가합니다."
                                />
                                <FeatureCard
                                    icon={<Cpu className="text-cyan-400" />}
                                    title="Tactical Extraction"
                                    description="수천 개의 데이터를 자동 클러스터링하여 팀 고유의 빌드업 및 공격 패턴을 핵심 전술로 요약합니다."
                                />
                                <FeatureCard
                                    icon={<Play className="text-emerald-400" />}
                                    title="Animated Visuals"
                                    description="추상적인 데이터 좌표를 고해상도 전술 애니메이션으로 변환하여 코칭 스태프와 선수들의 직관적 이해를 돕습니다."
                                />
                            </div>
                        </div>

                        <div className="flex items-center justify-center gap-10 mt-9">
                            <button
                                onClick={onAnalyze}
                                className="btn btn-primary text-4xl px-40 py-10 rounded-[40px] group font-bold shadow-[0_0_60px_rgba(99,102,241,0.6)] transition-all hover:scale-110 active:scale-95 border-2 border-white/20"
                            >
                                Analyze Tactics
                                <ArrowRight size={48} className="ml-6 group-hover:translate-x-3 transition-transform" />
                            </button>
                        </div>
                    </motion.div>
                </div>
            </section>

        </div>
    );
};

const FeatureCard = ({ icon, title, description }) => (
    <div className="glass-card">
        <div className="w-14 h-14 bg-white/5 rounded-2xl flex items-center justify-center mb-8 border border-white/10">
            {icon}
        </div>
        <h3 className="text-2xl mb-6">{title}</h3>
        <p className="text-slate-400 leading-relaxed font-light">
            {description}
        </p>
    </div>
);


export default Home;
