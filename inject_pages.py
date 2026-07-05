import sys

def main():
    file_path = "index.html"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pages = [
        {
            "id": "hair",
            "title": "Hair",
            "color": "yellow",
            "desc": "Upload an image of the scalp or hair condition. Our AI model analyzes 9 hair diseases including Alopecia, Folliculitis, and Pattern Baldness.",
            "icon": '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 2v13M12 2v13M17 2v13M4 15h16v7H4z"/></svg>',
            "result": "Alopecia Areata (92% Confidence)",
            "rec": "Consult a dermatologist. Avoid tight hairstyles."
        },
        {
            "id": "ct",
            "title": "CT Scan",
            "color": "blue",
            "desc": "Upload a CT scan image. Our model detects oncology, neurology, pulmonology, urology, and cardiology conditions.",
            "icon": '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><path d="M12 2v20M2 12h20"/></svg>',
            "result": "No major abnormalities detected (95% Confidence)",
            "rec": "Routine follow-up recommended."
        },
        {
            "id": "mri",
            "title": "MRI",
            "color": "indigo",
            "desc": "Upload an MRI scan. Our AI assists in neuro, MSK, and cardiac MRI analysis.",
            "icon": '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16v16H4zM4 12h16M12 4v16"/></svg>',
            "result": "Mild disc herniation L4-L5 (88% Confidence)",
            "rec": "Physical therapy recommended."
        },
        {
            "id": "ultrasound",
            "title": "Ultrasound",
            "color": "pink",
            "desc": "Upload an ultrasound image for obstetrics, abdominal, or superficial imaging analysis.",
            "icon": '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
            "result": "Normal fetal growth parameters (98% Confidence)",
            "rec": "Continue routine prenatal care."
        },
        {
            "id": "histopathology",
            "title": "Histopathology",
            "color": "emerald",
            "desc": "Upload histopathology slides for oncological, dermatopathology, and other pathological evaluations.",
            "icon": '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/></svg>',
            "result": "Benign cellular changes (91% Confidence)",
            "rec": "No signs of malignancy."
        }
    ]

    html_to_inject = "\n"
    for p in pages:
        html_to_inject += f"""
<!-- ================================ PAGE: {p['title'].upper()} AI ================================ -->
<main id="page-{p['id']}" class="page">
  <section class="pt-24 pb-12">
    <div class="max-w-4xl mx-auto px-5 lg:px-8 text-center">
      <div class="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-{p['color']}-500/15 text-{p['color']}-500 mb-6">
        {p['icon']}
      </div>
      <h1 class="font-display font-extrabold text-4xl mb-4">{p['title']} <span class="grad-text">Disease AI</span></h1>
      <p class="text-muted text-lg max-w-2xl mx-auto">{p['desc']}</p>
    </div>
  </section>
  <section class="pb-24">
    <div class="max-w-xl mx-auto px-5 lg:px-8">
      <div class="card rounded-3xl p-6 text-center">
        <div id="{p['id']}DropZone" class="p-10 text-center border-2 border-dashed border-[var(--card-border)] rounded-2xl cursor-pointer transition-colors" onclick="document.getElementById('{p['id']}FileInput').click()">
          <input type="file" id="{p['id']}FileInput" accept="image/*" class="hidden" onchange="handleSimulatedUpload('{p['id']}', event)">
          <div id="{p['id']}UploadPrompt">
            <svg class="mx-auto w-12 h-12 text-muted mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            <div class="font-semibold mb-1">Click or drag image here</div>
            <div class="text-muted text-sm">PNG, JPG up to 10MB</div>
          </div>
          <div id="{p['id']}ThumbWrap" class="hidden">
            <img id="{p['id']}Thumb" class="mx-auto rounded-xl max-h-48 object-cover" />
            <button onclick="event.stopPropagation(); clearSimulatedImage('{p['id']}')" class="mt-4 text-xs text-rose-400 hover:underline">Remove</button>
          </div>
        </div>
        <button id="{p['id']}AnalyseBtn" onclick="runSimulatedAnalysis('{p['id']}')" class="btn-primary w-full py-4 rounded-2xl font-bold mt-6 disabled:opacity-50" disabled>Analyse {p['title']}</button>
      </div>
      <div id="{p['id']}Result" class="hidden mt-6 card rounded-3xl p-6 border-{p['color']}-500 border-2 text-center">
        <div class="w-12 h-12 rounded-full bg-{p['color']}-500/20 text-{p['color']}-500 flex items-center justify-center mx-auto mb-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <h3 class="font-bold text-lg text-{p['color']}-500 mb-2">AI Analysis Complete</h3>
        <p class="text-sm font-semibold">{p['result']}</p>
        <p class="text-sm text-muted mt-2">Recommendation: {p['rec']}</p>
      </div>
    </div>
  </section>
</main>
"""

    js_to_inject = """
/* ===================== SIMULATED AI PAGES LOGIC ===================== */
function handleSimulatedUpload(id, event) {
  const file = event.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById(id + 'UploadPrompt').classList.add('hidden');
    document.getElementById(id + 'ThumbWrap').classList.remove('hidden');
    document.getElementById(id + 'Thumb').src = e.target.result;
    document.getElementById(id + 'AnalyseBtn').disabled = false;
    document.getElementById(id + 'Result').classList.add('hidden');
    document.getElementById(id + 'AnalyseBtn').innerHTML = 'Analyse Image';
  };
  reader.readAsDataURL(file);
}

function clearSimulatedImage(id) {
  document.getElementById(id + 'FileInput').value = '';
  document.getElementById(id + 'UploadPrompt').classList.remove('hidden');
  document.getElementById(id + 'ThumbWrap').classList.add('hidden');
  document.getElementById(id + 'AnalyseBtn').disabled = true;
  document.getElementById(id + 'Result').classList.add('hidden');
}

function runSimulatedAnalysis(id) {
  const btn = document.getElementById(id + 'AnalyseBtn');
  btn.disabled = true;
  btn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-current inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Analyzing...`;
  
  setTimeout(() => {
    btn.innerHTML = 'Analysis Complete';
    document.getElementById(id + 'Result').classList.remove('hidden');
  }, 2000);
}
"""

    target_html = "<!-- ================================ PAGE: LOGIN ================================ -->"
    if target_html in content:
        content = content.replace(target_html, html_to_inject + "\n" + target_html)
    else:
        print("Could not find PAGE: LOGIN marker")
        return

    target_js = "/* ===================== CHEST X-RAY AI LOGIC ===================== */"
    if target_js in content:
        content = content.replace(target_js, js_to_inject + "\n" + target_js)
    else:
        print("Could not find CHEST X-RAY AI LOGIC marker")
        return

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Injection successful.")

if __name__ == "__main__":
    main()
