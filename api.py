import os
import re
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from rag_tool import RAGDocumentTool, load_documents

load_dotenv()

app = FastAPI(
    title="Business Intelligence Agent API",
    description="Multi-Agent AI System untuk analisis bisnis otomatis",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

llm = LLM(
    model="openrouter/openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)

print("📂 Loading dokumen...")
load_documents("./documents")
print("✅ Siap!")

class AnalysisRequest(BaseModel):
    company_name: str
    industry: str

def run_analysis(company_name: str, industry: str) -> str:
    rag_tool = RAGDocumentTool()
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    doc_analyst = Agent(
        role="Internal Document Analyst",
        goal="Jawab pertanyaan tentang kebijakan dan informasi internal perusahaan",
        backstory=(
            "Analis dokumen internal yang teliti. "
            "Hanya menjawab berdasarkan dokumen resmi."
        ),
        tools=[rag_tool],
        verbose=False,
        llm=llm
    )

    competitor_analyst = Agent(
        role="Competitor Intelligence Analyst",
        goal="Kumpulkan info terbaru tentang kompetitor",
        backstory=(
            "Analis intelijen kompetitor berpengalaman. "
            "Mencari info dari internet secara real-time."
        ),
        tools=[search_tool, scrape_tool],
        verbose=False,
        llm=llm
    )

    report_writer = Agent(
        role="Business Intelligence Report Writer",
        goal="Buat laporan eksekutif yang jelas dan actionable",
        backstory=(
            "Penulis laporan bisnis senior yang mengubah "
            "data mentah menjadi insight yang mudah dipahami."
        ),
        tools=[rag_tool],
        verbose=False,
        llm=llm
    )

    internal_task = Task(
        description=(
            f"Analisis dokumen internal {company_name}:\n"
            "1. Kebijakan refund & support\n"
            "2. Struktur harga\n"
            "3. Keunggulan produk\n"
            "4. Target customer (ICP)"
        ),
        expected_output="Ringkasan internal perusahaan dengan sumber dokumen.",
        agent=doc_analyst,
        async_execution=True
    )

    competitor_task = Task(
        description=(
            f"Research kompetitor di industri {industry}:\n"
            "1. Cari 3 kompetitor utama\n"
            "2. Bandingkan harga & fitur\n"
            "3. Temukan kelemahan mereka"
        ),
        expected_output="Laporan kompetitor dengan perbandingan lengkap.",
        agent=competitor_analyst,
        async_execution=True
    )

    report_task = Task(
        description=(
            f"Buat Business Intelligence Report untuk {company_name}.\n"
            "PENTING: Tulis LANGSUNG laporan final dalam Markdown.\n"
            "DILARANG menulis thought, action, observation, atau JSON.\n"
            "LANGSUNG mulai dengan heading pertama.\n"
            "Format wajib:\n"
            "# Business Intelligence Report\n"
            "## 1. Executive Summary\n"
            "## 2. Internal Position\n"
            "## 3. Competitive Landscape\n"
            "## 4. Opportunities & Threats\n"
            "## 5. Recommendations\n\n"
            "Tulis dalam Bahasa Indonesia."
        ),
        expected_output=(
            "Laporan BI lengkap dalam format Markdown. "
            "Dimulai langsung dengan # heading. "
            "Tidak ada JSON, tidak ada thought/action/observation."
        ),
        context=[internal_task, competitor_task],
        agent=report_writer
    )

    crew = Crew(
        agents=[doc_analyst, competitor_analyst, report_writer],
        tasks=[internal_task, competitor_task, report_task],
        verbose=False
    )

    result = crew.kickoff(inputs={
        "company_name": company_name,
        "industry": industry
    })

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{company_name.replace(' ', '_')}_report.md"
    with open(filename, "w") as f:
        f.write(str(result))

    result_str = str(result)
    if "## " in result_str or "# " in result_str:
        match = re.search(r'#+ ', result_str)
        if match:
            result_str = result_str[match.start():]
    return result_str

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Business Intelligence Agent API",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "documents_loaded": True}

@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        print(f"📊 Mulai analisis: {request.company_name}")
        result = run_analysis(
            company_name=request.company_name,
            industry=request.industry
        )
        return JSONResponse(content={
            "status": "success",
            "company": request.company_name,
            "industry": request.industry,
            "report": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
