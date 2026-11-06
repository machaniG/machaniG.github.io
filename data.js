/* =====================================================================
   data.js  -  EVERYTHING YOU EDIT LIVES IN THIS ONE FILE
   Both index.html and project.html read from here.
   ===================================================================== */

/* ---------- 1. LOOK AND FEEL ---------- */
const THEME = "indigo";   // "amber" | "neon" | "sand" | "indigo" | "ocean" | "mono"   (see PALETTES at the bottom to tweak or add your own)
const MODE  = "light";   // first-visit mode: "dark", "light", or "auto" (follow the visitor's system). The ◐ button always lets visitors switch.

/* Visual effects: set any to false to switch it off */
const FX = {
  glass:      true,   // frosted, glowing cards and panels
  spot:       true,   // soft light that follows the cursor
  grad:       true,   // gradient colour on page headlines
  shimmer:    true,   // light sweep across primary buttons on hover
  typewriter: true    // rotating "I build ..." line in the hero
};
const ROLES     = ["forecasting models", "LLM and RAG apps", "MLOps pipelines", "dashboards people use"];  // words the hero line types out
const HAND_NOTE = "try moving your cursor";   // handwritten note beside the hero chart ("" to hide)

/* ---------- 2. YOUR CONTENT ----------
   PROJECTS are NOT in this file any more.
   Each project is one file in  content/projects/NAME.md  and its images go in  images/NAME/
   (see README.md). After editing, run:  python3 build.py
   Everything else (your name, services, experience, tools) is still edited below.
*/
const ME={
  name:"Your Name", initials:"YN", email:"you@example.com", photo:"", cv:"#",
  socials:[["GitHub","https://github.com/machaniG"],["LinkedIn","https://linkedin.com/in/yourusername"],["Kaggle","https://kaggle.com/yourusername"],["X","https://x.com/yourusername"]],
};
const SERVICES=[
 {id:"strategy",color:"var(--a)",title:"AI strategy and roadmap",blurb:"Find where AI will pay off in your business and where it won't.",
  gets:["Ranked list of use cases with cost and impact estimates","Data readiness report","A 6 to 12 month roadmap your team can follow"],time:"2 to 3 weeks",repo:"https://github.com/yourusername/ai-opportunity-scoring"},
 {id:"ml",color:"var(--c2)",title:"Predictive models and forecasting",blurb:"Churn, demand, pricing and risk models built for decisions, not leaderboards.",
  gets:["Baseline and tuned models with clear evaluation","Explainability so stakeholders trust the output","A tested pipeline that runs on new data"],time:"4 to 8 weeks",repo:"https://github.com/yourusername/demand-forecasting"},
 {id:"llm",color:"var(--c3)",title:"LLM and RAG applications",blurb:"Chat, search and document tools grounded in your own data.",
  gets:["Retrieval pipeline over your documents","Evaluation set to measure answer quality","Guardrails, cost controls and a simple front end"],time:"4 to 10 weeks",repo:"https://github.com/yourusername/rag-document-qa"},
 {id:"mlops",color:"var(--a)",title:"Data pipelines and MLOps",blurb:"Make models reliable: deployment, monitoring and retraining.",
  gets:["CI/CD for models and data","Drift and performance monitoring dashboards","Runbooks so your team can operate it"],time:"3 to 6 weeks",repo:"https://github.com/yourusername/mlops-template"}
];
/* PROJECTS now live in content/projects/*.md. Run  python3 build.py  after editing them. */
const EXPERIENCE=[
 {when:"2021 – now",role:"Freelance Data Scientist and AI Engineer",org:"Self-employed",pts:["Delivered 20+ projects across retail, legal and manufacturing","Led builds from data audit to production deployment"]},
 {when:"2018 – 2021",role:"Senior Data Scientist",org:"Company Name",pts:["Owned the forecasting platform used by 120 stores","Mentored three junior data scientists"]},
 {when:"2016 – 2018",role:"Data Analyst",org:"Company Name",pts:["Built dashboards and experiment analyses for the product team"]}
];
const EDU=[["MSc Data Science","University Name, 2016"],["BSc Statistics","University Name, 2014"],["AWS Certified Machine Learning, Specialty","2023"]];


/* TOOLS I USE: shown in groups on the home page. Rename groups, add or remove items freely. */
const TOOLS=[
 {group:"Languages",items:["Python","SQL","R","Bash"]},
 {group:"Machine learning and AI",items:["scikit-learn","PyTorch","XGBoost","LangChain","Hugging Face"]},
 {group:"Data and pipelines",items:["pandas","dbt","Airflow","Spark","BigQuery","Snowflake"]},
 {group:"Deployment and cloud",items:["Docker","AWS","GCP","FastAPI","MLflow","GitHub Actions"]},
 {group:"Analysis and sharing",items:["Streamlit","Jupyter","Tableau","Power BI"]}
];

/* WRITING / NOTES (optional): leave the list empty to hide the section. Example entry:
   {title:"Finding prime numbers with Python", date:"2021", tag:"Python", url:"https://..."} */
const WRITING=[];


/* ---------- 3. COLOUR PALETTES (edit freely) ----------
   bg = page background, surface = cards, ink = main text, muted = secondary text,
   line = borders, a = main accent (buttons, links), c2 / c3 = extra accents, grid = background pattern tint */
const PALETTES = {
  amber:{ // deep slate + warm orange (the orange from your old site)
    light:{bg:"#faf7f2",surface:"#ffffff",ink:"#1b1a17",muted:"#6a655c",line:"#e6dfd2",a:"#c2570a","a-ink":"#ffffff",c2:"#2563eb",c3:"#059669",grid:"rgba(194,87,10,.06)"},
    dark: {bg:"#0e1217",surface:"#161c24",ink:"#eef1f5",muted:"#98a2b1",line:"#252d38",a:"#ff9a2e","a-ink":"#1a1206",c2:"#38bdf8",c3:"#34d399",grid:"rgba(255,154,46,.06)"}},
  neon:{ // near-black violet + neon purple, cinematic and dark-first
    light:{bg:"#f5f2ff",surface:"#ffffff",ink:"#150d33",muted:"#5d5688",line:"#dcd4f7",a:"#6d28d9","a-ink":"#ffffff",c2:"#db2777",c3:"#0284c7",grid:"rgba(109,40,217,.07)",disp:"'Sora','Inter',system-ui,sans-serif"},
    dark: {bg:"#05010f",surface:"#0d0720",ink:"#f4f2ff",muted:"#a9a3c9",line:"#271a4d",a:"#a78bfa","a-ink":"#05010f",c2:"#f472b6",c3:"#38bdf8",grid:"rgba(139,92,246,.08)",disp:"'Sora','Inter',system-ui,sans-serif"}},
  sand:{ // warm cream + forest green
    light:{bg:"#f6f0e4",surface:"#fffaf0",ink:"#1d2a23",muted:"#646f63",line:"#e2d8c3",a:"#1f7a57","a-ink":"#ffffff",c2:"#d9552b",c3:"#2b59c3",grid:"rgba(31,122,87,.08)"},
    dark: {bg:"#121e19",surface:"#1a2a23",ink:"#f1ece0",muted:"#9fb0a5",line:"#2c3f35",a:"#5fd1a1","a-ink":"#0f1c16",c2:"#ff8a5c",c3:"#7ea2ff",grid:"rgba(95,209,161,.07)"}},
  indigo:{ // soft lavender + electric violet
    light:{bg:"#f3f1ff",surface:"#ffffff",ink:"#17143a",muted:"#5e5a85",line:"#dcd8f3",a:"#5b3df5","a-ink":"#ffffff",c2:"#ff4f8b",c3:"#14b8a6",grid:"rgba(91,61,245,.08)"},
    dark: {bg:"#0d0b24",surface:"#16133a",ink:"#ece9ff",muted:"#a29fd0",line:"#2a2660",a:"#9d8cff","a-ink":"#0d0b24",c2:"#ff7fae",c3:"#3ddbc8",grid:"rgba(157,140,255,.08)"}},
  ocean:{ // pale aqua + deep teal
    light:{bg:"#eaf6f7",surface:"#ffffff",ink:"#0b2a33",muted:"#4d6b73",line:"#cde4e7",a:"#0a7f95","a-ink":"#ffffff",c2:"#f26b3a",c3:"#6b4de6",grid:"rgba(10,127,149,.08)"},
    dark: {bg:"#08212a",surface:"#0e2f3a",ink:"#e6f6f8",muted:"#8fb4bc",line:"#1b4552",a:"#3cc7e0","a-ink":"#08212a",c2:"#ff8f63",c3:"#a28bff",grid:"rgba(60,199,224,.07)"}},
  mono:{ // off-white, black ink, one orange accent
    light:{bg:"#f3f3ef",surface:"#ffffff",ink:"#111111",muted:"#666660",line:"#dcdcd5",a:"#ff4d00","a-ink":"#ffffff",c2:"#111111",c3:"#0a8f6a",grid:"rgba(17,17,17,.06)"},
    dark: {bg:"#111111",surface:"#1b1b1b",ink:"#f2f2ee",muted:"#9a9a94",line:"#2e2e2e",a:"#ff6a2a","a-ink":"#111111",c2:"#f2f2ee",c3:"#3ddbb0",grid:"rgba(255,255,255,.05)"}}
};

/* ---------- 4. ENGINE: no need to edit below this line ---------- */
(function(){
  const P=PALETTES[THEME]||PALETTES.sand;
  const f=o=>Object.entries(o).map(([k,v])=>'--'+k+':'+v).join(';');
  const st=document.createElement('style');
  st.textContent=':root{'+f(P.light)+'}:root[data-theme=dark]{'+f(P.dark)+'}';
  document.head.appendChild(st);
  const R=document.documentElement;
  Object.keys(FX).forEach(k=>FX[k]&&R.classList.add('fx-'+k));
  if(FX.spot&&!matchMedia('(prefers-reduced-motion:reduce)').matches&&matchMedia('(hover:hover)').matches){
    let raf=0,x=0,y=0;
    addEventListener('pointermove',e=>{x=e.clientX;y=e.clientY;if(!raf)raf=requestAnimationFrame(()=>{raf=0;R.style.setProperty('--mx',x+'px');R.style.setProperty('--my',y+'px')})},{passive:true});
  }
  let saved=null;try{saved=localStorage.getItem('theme')}catch(e){}
  if(!saved&&MODE!=='auto')R.dataset.theme=MODE;
})();

function thumb(kind,seed){let s=seed;const r=()=>(s=(s*9301+49297)%233280)/233280;let o='<svg viewBox="0 0 200 100" preserveAspectRatio="none" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">';
 if(kind=='line'){let p=[];for(let i=0;i<14;i++)p.push((i*14.5)+','+(78-i*3.2-r()*16));o+='<polyline points="'+p.join(' ')+'"/><polyline stroke-dasharray="4 6" opacity=".4" points="'+p.map(q=>q.split(',')[0]+','+(+q.split(',')[1]+14)).join(' ')+'"/>'}
 else if(kind=='bars'){for(let i=0;i<12;i++){const h=30+r()*55;o+='<rect x="'+(i*16+4)+'" y="'+(96-h)+'" width="10" height="'+h+'" rx="3" fill="currentColor" stroke="none" opacity="'+(.35+r()*.65)+'"/>'}}
 else{const cs=[[50,30],[130,65],[80,80]];for(let i=0;i<48;i++){const c=cs[i%3];o+='<circle cx="'+(c[0]+(r()-.5)*56)+'" cy="'+(c[1]+(r()-.5)*36)+'" r="3" fill="'+['var(--a)','var(--c2)','var(--c3)'][i%3]+'" stroke="none"/>'}}
 return o+'</svg>'}
