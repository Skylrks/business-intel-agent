import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from rag_tool import RAGDocumentTool, load_documents

# =====================
# SETUP
# =====================
os.getenv("NAMA_KEY", "")

llm = LLM(
    model="openrouter/openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)

# Load dokumen ke ChromaDB saat startup
# Kalau dokumen sudah pernah di-load sebelumnya,
# upsert akan update yang berubah saja
print("📂 Loading dokumen internal...")
load_documents("./documents")
print("✅ Dokumen siap!\n")

# =====================
# TOOLS
# =====================
rag_tool = RAGDocumentTool()
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# =====================
# AGENTS
# =====================

# Agent 1: Jawab pertanyaan dari dokumen internal
doc_analyst = Agent(
    role="Internal Document Analyst",
    goal=(
        "Jawab pertanyaan tentang kebijakan, prosedur, "
        "dan informasi internal perusahaan dengan akurat "
        "berdasarkan dokumen yang tersedia."
    ),
    backstory=(
        "Kamu adalah analis dokumen internal yang sangat teliti. "
        "Kamu hanya menjawab berdasarkan dokumen resmi perusahaan "
        "dan selalu menyebutkan sumber informasinya. "
        "Jika informasi tidak ada di dokumen, kamu terus terang bilang tidak tahu."
    ),
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

# Agent 2: Research kompetitor dari internet
competitor_analyst = Agent(
    role="Competitor Intelligence Analyst",
    goal=(
        "Kumpulkan dan analisis informasi terbaru tentang "
        "kompetitor untuk memberikan insight strategis."
    ),
    backstory=(
        "Kamu adalah analis intelijen kompetitor berpengalaman. "
        "Kamu mencari informasi terbaru dari internet tentang "
        "produk, harga, dan strategi kompetitor, lalu "
        "menyajikannya dalam format yang actionable."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

# Agent 3: Tulis laporan final
report_writer = Agent(
    role="Business Intelligence Report Writer",
    goal=(
        "Gabungkan semua insight menjadi laporan eksekutif "
        "yang jelas, terstruktur, dan actionable."
    ),
    backstory=(
        "Kamu adalah penulis laporan bisnis senior yang ahli "
        "mengubah data mentah dan analisis teknis menjadi "
        "laporan yang mudah dipahami eksekutif. "
        "Laporanmu selalu punya rekomendasi yang konkret."
    ),
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

# Agent 4: QA - pastikan laporan akurat
qa_reviewer = Agent(
    role="Quality Assurance Reviewer",
    goal=(
        "Pastikan laporan final akurat, lengkap, "
        "dan bebas dari informasi yang tidak terverifikasi."
    ),
    backstory=(
        "Kamu adalah QA reviewer yang sangat detail-oriented. "
        "Kamu cross-check setiap klaim di laporan dengan "
        "sumber aslinya dan memastikan tidak ada asumsi "
        "yang tidak berdasar."
    ),
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

# =====================
# TASKS
# =====================

# Task 1: Analisis dokumen internal (async - paralel dengan task 2)
internal_analysis_task = Task(
    description=(
        "Analisis dokumen internal perusahaan {company_name} dan jawab:\n"
        "1. Apa kebijakan refund dan support yang berlaku?\n"
        "2. Apa saja paket harga yang tersedia?\n"
        "3. Apa keunggulan produk dibanding kompetitor?\n"
        "4. Apa target customer ideal (ICP) perusahaan?\n\n"
        "Gunakan RAG tool untuk mencari di dokumen. "
        "Sertakan sumber untuk setiap informasi."
    ),
    expected_output=(
        "Ringkasan terstruktur berisi: kebijakan perusahaan, "
        "struktur harga, keunggulan produk, dan ICP — "
        "semuanya dengan referensi sumber dokumen."
    ),
    agent=doc_analyst,
    async_execution=True  # Jalan paralel dengan task 2
)

# Task 2: Research kompetitor (async - paralel dengan task 1)
competitor_research_task = Task(
    description=(
        "Lakukan research mendalam tentang kompetitor "
        "di industri {industry}:\n"
        "1. Cari 3 kompetitor utama\n"
        "2. Bandingkan harga dan fitur mereka\n"
        "3. Cari kelemahan atau complaint pelanggan mereka\n"
        "4. Identifikasi peluang diferensiasi\n\n"
        "Fokus pada informasi yang dipublish dalam 3 bulan terakhir."
    ),
    expected_output=(
        "Laporan kompetitor berisi: daftar kompetitor, "
        "perbandingan harga/fitur, kelemahan kompetitor, "
        "dan peluang yang bisa dimanfaatkan."
    ),
    agent=competitor_analyst,
    async_execution=True  # Jalan paralel dengan task 1
)

# Task 3: Tulis laporan (tunggu task 1 & 2 selesai)
report_writing_task = Task(
    description=(
        "Buat laporan Business Intelligence mingguan untuk {company_name}.\n\n"
        "PENTING: Langsung tulis laporan final dalam Markdown.\n"
        "JANGAN tulis thought, action, atau observation.\n\n"
        "Gunakan hasil dari:\n"
        "- Analisis dokumen internal (internal strengths)\n"
        "- Research kompetitor (external landscape)\n\n"
        "Format wajib:\n"
        "# Business Intelligence Report - {company_name}\n"
        "## 1. Executive Summary\n"
        "## 2. Internal Position\n"
        "## 3. Competitive Landscape\n"
        "## 4. Opportunities & Threats\n"
        "## 5. Recommendations\n\n"
        "Tulis dalam Bahasa Indonesia. Langsung mulai dengan # heading."
    ),
    expected_output=(
        "Laporan BI lengkap dalam format Markdown. "
        "Dimulai langsung dengan # heading. "
        "Tidak ada JSON, tidak ada thought/action/observation."
    ),
    context=[internal_analysis_task, competitor_research_task],
    output_file="reports/weekly_report.md",
    agent=report_writer
)
# Task 4: QA review laporan
qa_task = Task(
    description=(
        "Review laporan yang dibuat oleh Report Writer:\n"
        "1. Verifikasi klaim tentang internal perusahaan "
        "   dengan RAG tool\n"
        "2. Tandai informasi yang tidak bisa diverifikasi\n"
        "3. Pastikan semua rekomendasi logis dan berdasar\n"
        "4. Perbaiki atau flag bagian yang perlu dikoreksi\n\n"
        "Output: laporan yang sudah diverifikasi dan dipoles."
    ),
    expected_output=(
        "Laporan final yang sudah diverifikasi, "
        "dengan catatan QA jika ada yang perlu diperhatikan."
    ),
    context=[report_writing_task],
    output_file="reports/final_report.md",
    agent=qa_reviewer
)

# =====================
# CREW
# =====================
crew = Crew(
    agents=[doc_analyst, competitor_analyst, report_writer, qa_reviewer],
    tasks=[
        internal_analysis_task,
        competitor_research_task,
        report_writing_task,
        qa_task
    ],
    verbose=True
)

# =====================
# JALANKAN
# =====================
inputs = {
    "company_name": "TechVision Inc",
    "industry": "B2B SaaS project management software"
}

print("🚀 Business Intelligence Agent mulai bekerja...")
print(f"🏢 Perusahaan: {inputs['company_name']}")
print(f"🏭 Industri: {inputs['industry']}")
print("-" * 50)

result = crew.kickoff(inputs=inputs)

print("\n\n✅ SELESAI!")
print("📄 Laporan tersimpan di: reports/final_report.md")
print("\n===== PREVIEW =====")
print(str(result)[:500] + "...")