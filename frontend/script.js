document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const clearImageBtn = document.getElementById('clearImageBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictSpinner = document.getElementById('predictSpinner');
    
    const emptyState = document.getElementById('emptyState');
    const resultState = document.getElementById('resultState');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    
    const steeringWheel = document.getElementById('steeringWheel');
    const angleValue = document.getElementById('angleValue');
    const lanesDetectedText = document.getElementById('lanesDetectedText');
    
    const dirLeft = document.getElementById('dirLeft');
    const dirStraight = document.getElementById('dirStraight');
    const dirRight = document.getElementById('dirRight');
    
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    
    // New Elements for Object Detection
    const odToggle = document.getElementById('odToggle');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const objectsContainer = document.getElementById('objectsContainer');
    const objectsList = document.getElementById('objectsList');
    
    // Decision Breakdown Elements
    const uiLaneSuggest = document.getElementById('uiLaneSuggest');
    const uiObsInfluence = document.getElementById('uiObsInfluence');
    const uiFinalDecision = document.getElementById('uiFinalDecision');
    const uiDecisionReason = document.getElementById('uiDecisionReason');
    const decisionReasonBox = document.getElementById('decisionReasonBox');
    
    let currentFile = null;
    let chartInstance = null;

    // --- Theme Toggle ---
    const toggleTheme = () => {
        document.documentElement.classList.toggle('dark');
        if (document.documentElement.classList.contains('dark')) {
            themeIcon.classList.replace('fa-moon', 'fa-sun');
        } else {
            themeIcon.classList.replace('fa-sun', 'fa-moon');
        }
        renderChart(); // Re-render chart for colors
    };
    themeToggle.addEventListener('click', toggleTheme);

    // --- File Upload Logic ---
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary', 'bg-blue-50', 'dark:bg-blue-900/20');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary', 'bg-blue-50', 'dark:bg-blue-900/20');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary', 'bg-blue-50', 'dark:bg-blue-900/20');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    const handleFile = (file) => {
        if (!file.type.startsWith('image/')) {
            showError("Please upload a valid image file.");
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
            hideError();
        };
        reader.readAsDataURL(file);
    };

    clearImageBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        predictBtn.disabled = true;
        hideResults();
        hideError();
        
        // Clear canvas and OD UI
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        objectsContainer.classList.add('hidden');
        uiLaneSuggest.innerText = '-';
        uiObsInfluence.innerText = '-';
        uiFinalDecision.innerText = '-';
        uiDecisionReason.innerText = 'Waiting for prediction...';
    });

    // --- Prediction Logic ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        predictBtn.disabled = true;
        predictSpinner.classList.remove('hidden');
        hideError();
        hideResults();
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('enable_od', odToggle.checked);
        
        try {
            // API call to backend (assumes backend is running on same host)
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || "Failed to predict steering angle.");
            }
            
            showResults(data);
            
        } catch (error) {
            showError(error.message);
        } finally {
            predictBtn.disabled = false;
            predictSpinner.classList.add('hidden');
        }
    });

    const showResults = (data) => {
        emptyState.classList.add('hidden');
        resultState.classList.remove('hidden');
        
        // Update Angle Text
        const angle = data.steering_angle;
        angleValue.innerText = angle.toFixed(3) + "°";
        
        // Rotate Steering Wheel (scale angle for visual effect)
        // Let's assume prediction is roughly between -0.2 to 0.2
        const visualRotation = angle * 200; // Multiplier to make it look prominent
        steeringWheel.style.transform = `rotate(${visualRotation}deg)`;
        
        // Update Direction Indicators based on final decision
        dirLeft.className = "dir-indicator w-16 h-16 rounded-xl bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-400 text-3xl shadow-inner transition-all duration-300";
        dirStraight.className = "dir-indicator w-16 h-16 rounded-xl bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-400 text-3xl shadow-inner transition-all duration-300";
        dirRight.className = "dir-indicator w-16 h-16 rounded-xl bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-400 text-3xl shadow-inner transition-all duration-300";
        
        let dirToDisplay = data.final_decision || data.direction || data.lane_direction;
        
        // Populate Decision Breakdown
        uiLaneSuggest.innerText = data.lane_direction || data.direction || 'Straight';
        
        let obsText = "None";
        if (data.distance_estimation && data.distance_estimation.length > 0) {
            const influences = data.distance_estimation.map(o => `${o.position} (${o.distance})`);
            const uniqueInfluences = [...new Set(influences)];
            obsText = uniqueInfluences.join(', ');
        }
        uiObsInfluence.innerText = obsText;
        
        let finalDecText = dirToDisplay.toUpperCase();
        if (dirToDisplay === 'stop') finalDecText = '🔴 STOP';
        else if (dirToDisplay === 'slow down') finalDecText = '🟡 SLOW DOWN';
        else finalDecText = '🟢 SAFE (' + finalDecText + ')';
        
        uiFinalDecision.innerText = finalDecText;
        uiDecisionReason.innerText = data.decision_reason || "Decision made based on current inputs.";
        
        // Update styling of the reason box based on final decision severity
        decisionReasonBox.className = "bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 text-green-800 dark:text-green-200 p-3 rounded text-sm transition-colors duration-300";
        if (dirToDisplay === 'stop') {
             decisionReasonBox.className = "bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 text-red-800 dark:text-red-200 p-3 rounded text-sm transition-colors duration-300";
        } else if (dirToDisplay === 'slow down') {
             decisionReasonBox.className = "bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 text-yellow-800 dark:text-yellow-200 p-3 rounded text-sm transition-colors duration-300";
        } else if (data.final_decision !== data.lane_direction) {
             decisionReasonBox.className = "bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 text-blue-800 dark:text-blue-200 p-3 rounded text-sm transition-colors duration-300";
        }
        
        let steerIndicator = data.direction || 'straight';
        if (steerIndicator === 'left') {
            dirLeft.classList.add('dir-active-left');
        } else if (steerIndicator === 'right') {
            dirRight.classList.add('dir-active-right');
        } else {
            dirStraight.classList.add('dir-active-straight');
        }
        
        // --- Render Object Detection ---
        const ctx = overlayCanvas.getContext('2d');
        overlayCanvas.width = imagePreview.clientWidth;
        overlayCanvas.height = imagePreview.clientHeight;
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        // Calculate scaling factors for object-contain
        const imgNaturalWidth = imagePreview.naturalWidth;
        const imgNaturalHeight = imagePreview.naturalHeight;
        
        let scaleX = 1;
        let scaleY = 1;
        let offsetX = 0;
        let offsetY = 0;
        
        if (imgNaturalWidth && imgNaturalHeight) {
            const displayRatio = overlayCanvas.width / overlayCanvas.height;
            const naturalRatio = imgNaturalWidth / imgNaturalHeight;
            
            if (naturalRatio > displayRatio) {
                // Constrained by width
                scaleX = overlayCanvas.width / imgNaturalWidth;
                scaleY = scaleX;
                offsetY = (overlayCanvas.height - (imgNaturalHeight * scaleY)) / 2;
            } else {
                // Constrained by height
                scaleY = overlayCanvas.height / imgNaturalHeight;
                scaleX = scaleY;
                offsetX = (overlayCanvas.width - (imgNaturalWidth * scaleX)) / 2;
            }
        }

        objectsList.innerHTML = '';
        if (data.objects_detected && data.objects_detected.length > 0) {
            objectsContainer.classList.remove('hidden');
            data.objects_detected.forEach(obj => {
                const [bx, by, bw, bh] = obj.bbox;
                const finalX = offsetX + (bx * scaleX);
                const finalY = offsetY + (by * scaleY);
                const finalW = bw * scaleX;
                const finalH = bh * scaleY;
                
                // Draw Box
                ctx.strokeStyle = '#ef4444'; // Red
                ctx.lineWidth = 2;
                ctx.strokeRect(finalX, finalY, finalW, finalH);
                
                // Draw Label
                ctx.fillStyle = '#ef4444';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(`${obj.label} ${(obj.confidence*100).toFixed(0)}%`, finalX, finalY > 15 ? finalY - 5 : finalY + 15);
                
                // Add to list
                const li = document.createElement('li');
                li.className = 'flex justify-between items-center text-sm p-2 bg-white dark:bg-gray-700 rounded shadow-sm';
                li.innerHTML = `
                    <div class="flex items-center gap-2 capitalize">
                        <i class="fa-solid fa-cube text-primary"></i>
                        <span>${obj.label} <span class="text-xs text-gray-400">(${obj.position})</span></span>
                    </div>
                    <span class="font-semibold text-green-500">${(obj.confidence*100).toFixed(0)}%</span>
                `;
                objectsList.appendChild(li);
            });
        } else {
            objectsContainer.classList.add('hidden');
        }
        
        lanesDetectedText.innerText = `Lanes detected: ${data.detected_lanes}`;
    };

    const hideResults = () => {
        emptyState.classList.remove('hidden');
        resultState.classList.add('hidden');
        steeringWheel.style.transform = `rotate(0deg)`;
    };

    const showError = (msg) => {
        errorMessage.innerText = msg;
        errorAlert.classList.remove('hidden');
    };

    const hideError = () => {
        errorAlert.classList.add('hidden');
    };

    // --- Feature Importance Chart ---
    let featureData = null;

    const fetchFeatureImportance = async () => {
        try {
            const res = await fetch('http://localhost:8000/feature-importance');
            if (res.ok) {
                featureData = await res.json();
                renderChart();
            }
        } catch (e) {
            console.error("Could not load feature importance", e);
        }
    };

    const renderChart = () => {
        if (!featureData) return;
        
        const isDark = document.documentElement.classList.contains('dark');
        const textColor = isDark ? '#f1f5f9' : '#334155';
        const gridColor = isDark ? '#475569' : '#e2e8f0';

        const labels = Object.keys(featureData);
        const values = Object.values(featureData);

        const ctx = document.getElementById('featureChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importance Score',
                    data: values,
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: textColor }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    };

    // Load initial data
    fetchFeatureImportance();
});
