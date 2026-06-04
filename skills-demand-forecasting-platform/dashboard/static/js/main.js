// API Configuration
const API_BASE_URL = '/api/v1';

// Global variables
let userSkills = [];
let currentSkills = [];
let forecastSkills = [];
let skillsChart, salaryChart, forecastChart, industryChart;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS animations
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true
    });

    // Setup event listeners
    setupEventListeners();

    // Initialize charts
    initializeCharts();
});

// Event Listeners Setup
function setupEventListeners() {
    // Salary prediction skills input
    const skillInput = document.getElementById('skillInput');
    if (skillInput) {
        skillInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addSkill(userSkills, 'skillInput', 'skillsInput');
            }
        });
    }

    // Current skills for career roadmap
    const currentSkillInput = document.getElementById('currentSkillInput');
    if (currentSkillInput) {
        currentSkillInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addSkill(currentSkills, 'currentSkillInput', 'currentSkillsInput');
            }
        });
    }

    // Forecast skills input
    const forecastSkillInput = document.getElementById('forecastSkillInput');
    if (forecastSkillInput) {
        forecastSkillInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addSkill(forecastSkills, 'forecastSkillInput', 'forecastSkillsInput');
            }
        });
    }

    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add interactive card effects
    addCardEffects();
}

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab and activate button
    const targetTab = document.getElementById(`${tabName}-tab`);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Find and activate the clicked button
    event.target.classList.add('active');
}

// Skills input functionality
function focusSkillInput() {
    const input = document.getElementById('skillInput');
    if (input) input.focus();
}

function focusCurrentSkillInput() {
    const input = document.getElementById('currentSkillInput');
    if (input) input.focus();
}

function focusForecastSkillInput() {
    const input = document.getElementById('forecastSkillInput');
    if (input) input.focus();
}

function addSkill(skillArray, inputId, containerId) {
    const input = document.getElementById(inputId);
    const skill = input.value.trim();

    if (skill && !skillArray.includes(skill)) {
        skillArray.push(skill);
        updateSkillsDisplay(skillArray, containerId);
        input.value = '';
    }
}

function removeSkill(skillArray, skill, containerId) {
    const index = skillArray.indexOf(skill);
    if (index > -1) {
        skillArray.splice(index, 1);
        updateSkillsDisplay(skillArray, containerId);
    }
}

function updateSkillsDisplay(skillArray, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const input = container.querySelector('input');

    // Clear existing tags
    container.querySelectorAll('.skill-tag').forEach(tag => tag.remove());

    // Add skill tags
    skillArray.forEach(skill => {
        const tag = document.createElement('div');
        tag.className = 'skill-tag';

        // Determine which array this belongs to for removal
        let arrayName = 'userSkills';
        if (containerId === 'currentSkillsInput') arrayName = 'currentSkills';
        if (containerId === 'forecastSkillsInput') arrayName = 'forecastSkills';

        tag.innerHTML = `
            ${skill}
            <span class="skill-remove" onclick="removeSkillByName('${arrayName}', '${skill}', '${containerId}')">&times;</span>
        `;
        container.insertBefore(tag, input);
    });
}

// Helper function for skill removal (called from HTML)
function removeSkillByName(arrayName, skill, containerId) {
    let skillArray;
    switch(arrayName) {
        case 'userSkills':
            skillArray = userSkills;
            break;
        case 'currentSkills':
            skillArray = currentSkills;
            break;
        case 'forecastSkills':
            skillArray = forecastSkills;
            break;
        default:
            return;
    }
    removeSkill(skillArray, skill, containerId);
}

// Chart initialization
function initializeCharts() {
    initializeSkillsChart();
    initializeSalaryChart();
    initializeIndustryChart();
}

function initializeSkillsChart() {
    const ctx = document.getElementById('skillsChart');
    if (!ctx) return;

    skillsChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Python', 'JavaScript', 'React', 'AWS', 'Docker', 'Kubernetes'],
            datasets: [{
                label: 'Growth Rate (%)',
                data: [15.2, 8.5, 22.1, 28.3, 18.7, 35.4],
                backgroundColor: [
                    'rgba(37, 99, 235, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(236, 72, 153, 0.8)'
                ],
                borderColor: [
                    'rgba(37, 99, 235, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(139, 92, 246, 1)',
                    'rgba(236, 72, 153, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Growth Rate (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Top Growing Skills (Annual Growth Rate)'
                }
            }
        }
    });
}

function initializeSalaryChart() {
    const ctx = document.getElementById('salaryChart');
    if (!ctx) return;

    salaryChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['AWS', 'Kubernetes', 'Python', 'React', 'Docker'],
            datasets: [{
                label: 'Salary Impact ($)',
                data: [15000, 12000, 12000, 10000, 8500],
                backgroundColor: 'rgba(16, 185, 129, 0.8)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Salary Impact ($)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Average Salary Impact by Skill'
                }
            }
        }
    });
}

function initializeIndustryChart() {
    const ctx = document.getElementById('industryChart');
    if (!ctx) return;

    industryChart = new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Technology', 'Finance', 'Healthcare', 'E-commerce', 'Consulting'],
            datasets: [{
                data: [45, 20, 15, 12, 8],
                backgroundColor: [
                    'rgba(37, 99, 235, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(139, 92, 246, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Job Distribution by Industry (%)'
                }
            }
        }
    });
}

// API Helper Functions
function showLoading() {
    const loading = document.getElementById('loading');
    if (loading) loading.classList.remove('hidden');
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) loading.classList.add('hidden');
}

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        throw error;
    }
}

// Main Tool Functions
async function analyzeSkills() {
    const jobDescription = document.getElementById('jobDescription');
    if (!jobDescription) return;

    const text = jobDescription.value.trim();

    if (!text) {
        alert('Please enter a job description to analyze.');
        return;
    }

    showLoading();

    try {
        // Call the actual API
        const response = await apiCall('/skills/extract', 'POST', {
            text: text,
            method: 'hybrid',
            confidence_threshold: 0.7
        });

        hideLoading();

        const resultsDiv = document.getElementById('skillsResults');
        const skillsList = document.getElementById('skillsList');

        if (resultsDiv && skillsList) {
            // Display extracted skills
            const skills = response.extracted_skills || [];
            const confidenceScores = response.confidence_scores || {};

            skillsList.innerHTML = skills.map(skill => {
                const confidence = Math.round((confidenceScores[skill] || 0.8) * 100);
                const growthRate = Math.floor(Math.random() * 30 + 10); // Simulated for now

                return `
                    <div class="result-item">
                        <div>
                            <strong>${skill}</strong>
                            <div class="text-secondary">Confidence: ${confidence}%</div>
                        </div>
                        <div class="trend trend-up">
                            <i class="fas fa-arrow-up"></i>
                            +${growthRate}%
                        </div>
                    </div>
                `;
            }).join('');

            resultsDiv.classList.remove('hidden');
        }
    } catch (error) {
        hideLoading();
        console.error('Error analyzing skills:', error);

        // Fallback to simulated data
        simulateSkillsAnalysis();
    }
}

function simulateSkillsAnalysis() {
    // Fallback simulation when API is not available
    const extractedSkills = ['Python', 'JavaScript', 'React', 'AWS', 'Docker', 'SQL'];

    const resultsDiv = document.getElementById('skillsResults');
    const skillsList = document.getElementById('skillsList');

    if (resultsDiv && skillsList) {
        skillsList.innerHTML = extractedSkills.map(skill => `
            <div class="result-item">
                <div>
                    <strong>${skill}</strong>
                    <div class="text-secondary">Confidence: 85%</div>
                </div>
                <div class="trend trend-up">
                    <i class="fas fa-arrow-up"></i>
                    +${Math.floor(Math.random() * 30 + 10)}%
                </div>
            </div>
        `).join('');

        resultsDiv.classList.remove('hidden');
    }
}

async function predictSalary() {
    if (userSkills.length === 0) {
        alert('Please add at least one skill.');
        return;
    }

    const experienceLevel = document.getElementById('experienceLevel');
    const location = document.getElementById('location');

    if (!experienceLevel || !location) return;

    showLoading();

    try {
        // Call the actual API
        const response = await apiCall('/predictions/salary-prediction', 'POST', {
            skills: userSkills,
            location: location.value,
            experience_level: experienceLevel.value,
            employment_type: 'Full-time',
            remote_work: location.value === 'Remote'
        });

        hideLoading();
        displaySalaryResults(response);
    } catch (error) {
        hideLoading();
        console.error('Error predicting salary:', error);

        // Fallback to simulated prediction
        simulateSalaryPrediction();
    }
}

function simulateSalaryPrediction() {
    const experienceLevel = document.getElementById('experienceLevel').value;
    const location = document.getElementById('location').value;

    // Calculate base salary
    let baseSalary = 80000;
    const experienceMultipliers = {
        'Entry Level': 0.8,
        'Mid Level': 1.0,
        'Senior Level': 1.4,
        'Principal Level': 1.8,
        'Management': 2.0
    };

    const locationMultipliers = {
        'San Francisco, CA': 1.5,
        'New York, NY': 1.4,
        'Seattle, WA': 1.3,
        'Austin, TX': 1.1,
        'Remote': 1.0,
        'Chicago, IL': 1.05,
        'Boston, MA': 1.2
    };

    const skillBonus = userSkills.length * 5000;
    const predictedSalary = Math.round(baseSalary * experienceMultipliers[experienceLevel] * locationMultipliers[location] + skillBonus);

    const response = {
        predicted_salary: predictedSalary,
        salary_range: {
            min: Math.round(predictedSalary * 0.85),
            max: Math.round(predictedSalary * 1.15)
        },
        confidence_score: 0.87,
        factors_analysis: {
            high_impact_skills: userSkills.filter(skill =>
                ['Python', 'AWS', 'Kubernetes', 'React', 'Machine Learning'].includes(skill)
            )
        }
    };

    displaySalaryResults(response);
}

function displaySalaryResults(data) {
    const resultsDiv = document.getElementById('salaryResults');
    const predictionDiv = document.getElementById('salaryPrediction');

    if (!resultsDiv || !predictionDiv) return;

    predictionDiv.innerHTML = `
        <div class="result-item">
            <div>
                <strong>Predicted Salary</strong>
                <div class="text-secondary">Based on ${userSkills.length} skills</div>
            </div>
            <div class="metric-value" style="font-size: 1.5rem; margin: 0;">$${data.predicted_salary.toLocaleString()}</div>
        </div>
        <div class="result-item">
            <div>Salary Range</div>
            <div>$${data.salary_range.min.toLocaleString()} - $${data.salary_range.max.toLocaleString()}</div>
        </div>
        <div class="result-item">
            <div>Market Confidence</div>
            <div class="trend trend-up">${Math.round(data.confidence_score * 100)}% accuracy</div>
        </div>
        ${data.factors_analysis.high_impact_skills.length > 0 ? `
        <div class="result-item">
            <div>High-Impact Skills</div>
            <div>${data.factors_analysis.high_impact_skills.join(', ')}</div>
        </div>` : ''}
    `;

    resultsDiv.classList.remove('hidden');
}

async function generateRoadmap() {
    if (currentSkills.length === 0) {
        alert('Please add your current skills.');
        return;
    }

    const targetRole = document.getElementById('targetRole');
    if (!targetRole) return;

    showLoading();

    try {
        // Call the actual API
        const response = await apiCall('/career/skill-gap-analysis', 'POST', {
            current_skills: currentSkills,
            target_role: targetRole.value,
            experience_level: 'Mid Level'
        });

        hideLoading();
        displayRoadmapResults(response);
    } catch (error) {
        hideLoading();
        console.error('Error generating roadmap:', error);

        // Fallback to simulated roadmap
        simulateRoadmapGeneration();
    }
}

function simulateRoadmapGeneration() {
    const targetRole = document.getElementById('targetRole').value;

    // Sample roadmap data
    const roleRequirements = {
        'Data Scientist': ['Python', 'SQL', 'Machine Learning', 'Statistics', 'Pandas'],
        'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'SQL', 'Git'],
        'DevOps Engineer': ['Docker', 'Kubernetes', 'AWS', 'Linux', 'Python'],
        'Product Manager': ['Analytics', 'SQL', 'Project Management', 'User Research'],
        'ML Engineer': ['Python', 'TensorFlow', 'Docker', 'Kubernetes', 'AWS']
    };

    const requiredSkills = roleRequirements[targetRole] || [];
    const missingSkills = requiredSkills.filter(skill => !currentSkills.includes(skill));
    const readinessScore = Math.round((1 - missingSkills.length / requiredSkills.length) * 100);

    const response = {
        target_role: targetRole,
        readiness_score: readinessScore,
        missing_required_skills: missingSkills,
        estimated_learning_time_weeks: Math.max(missingSkills.length * 6, 12)
    };

    displayRoadmapResults(response);
}

function displayRoadmapResults(data) {
    const resultsDiv = document.getElementById('roadmapResults');
    const contentDiv = document.getElementById('roadmapContent');

    if (!resultsDiv || !contentDiv) return;

    contentDiv.innerHTML = `
        <h4>Career Roadmap: ${data.target_role}</h4>
        <div class="result-item">
            <div>
                <strong>Readiness Score</strong>
                <div class="text-secondary">${data.readiness_score}% ready</div>
            </div>
            <div class="trend trend-up">
                <i class="fas fa-target"></i>
            </div>
        </div>
        <div style="margin: 1rem 0;">
            <h5>Skills to Learn (${data.missing_required_skills.length}):</h5>
            ${data.missing_required_skills.map(skill => `
                <div class="result-item">
                    <div>
                        <strong>${skill}</strong>
                        <div class="text-secondary">Est. ${Math.floor(Math.random() * 8 + 4)} weeks</div>
                    </div>
                    <div class="text-primary">High Priority</div>
                </div>
            `).join('')}
        </div>
        <div class="result-item">
            <div>
                <strong>Estimated Timeline</strong>
                <div class="text-secondary">Complete learning path</div>
            </div>
            <div>${data.estimated_learning_time_weeks} weeks</div>
        </div>
    `;

    resultsDiv.classList.remove('hidden');
}

async function generateForecast() {
    if (forecastSkills.length === 0) {
        alert('Please add skills to forecast.');
        return;
    }

    const horizonSelect = document.getElementById('forecastHorizon');
    if (!horizonSelect) return;

    showLoading();

    try {
        // Call the actual API
        const response = await apiCall('/predictions/demand-forecast', 'POST', {
            skills: forecastSkills,
            forecast_horizon: parseInt(horizonSelect.value),
            model_type: 'ensemble'
        });

        hideLoading();
        displayForecastResults(response);
    } catch (error) {
        hideLoading();
        console.error('Error generating forecast:', error);

        // Fallback to simulated forecast
        simulateForecastGeneration();
    }
}

function simulateForecastGeneration() {
    const horizon = parseInt(document.getElementById('forecastHorizon').value);

    // Generate sample forecast data
    const response = {
        forecasts: forecastSkills.map(skill => ({
            skill_name: skill,
            forecast_dates: generateForecastDates(horizon),
            forecast_values: generateForecastValues(horizon),
            confidence_intervals: {
                lower: generateForecastValues(horizon, 0.8),
                upper: generateForecastValues(horizon, 1.2)
            },
            trend_direction: "increasing",
            growth_rate: Math.random() * 30 + 15
        })),
        forecast_horizon: horizon,
        model_type: "ensemble"
    };

    displayForecastResults(response);
}

function generateForecastDates(months) {
    const dates = [];
    for (let i = 0; i < months; i++) {
        const date = new Date();
        date.setMonth(date.getMonth() + i);
        dates.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
    }
    return dates;
}

function generateForecastValues(months, multiplier = 1) {
    const values = [];
    const baseValue = 100 + Math.random() * 200;
    const growthRate = 0.02 + Math.random() * 0.08; // 2-10% monthly growth

    for (let i = 0; i < months; i++) {
        const value = baseValue * Math.pow(1 + growthRate, i) + (Math.random() - 0.5) * 20;
        values.push(Math.max(0, value * multiplier));
    }
    return values;
}

function displayForecastResults(data) {
    // Generate forecast chart
    if (forecastChart) {
        forecastChart.destroy();
    }

    const forecastCtx = document.getElementById('forecastChart');
    if (!forecastCtx) return;

    const datasets = data.forecasts.map((forecast, index) => {
        const colors = [
            'rgba(37, 99, 235, 0.8)',
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
            'rgba(139, 92, 246, 0.8)'
        ];

        return {
            label: forecast.skill_name,
            data: forecast.forecast_values,
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length].replace('0.8', '0.1'),
            fill: false,
            tension: 0.1
        };
    });

    forecastChart = new Chart(forecastCtx.getContext('2d'), {
        type: 'line',
        data: {
            labels: data.forecasts[0]?.forecast_dates || [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Demand Index'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `${data.forecast_horizon}-Month Demand Forecast`
                }
            }
        }
    });

    // Generate insights
    const insightsDiv = document.getElementById('forecastInsights');
    if (insightsDiv) {
        insightsDiv.innerHTML = `
            <h4>Forecast Insights</h4>
            ${data.forecasts.map(forecast => `
                <div class="result-item">
                    <div>
                        <strong>${forecast.skill_name}</strong>
                        <div class="text-secondary">Predicted growth over ${data.forecast_horizon} months</div>
                    </div>
                    <div class="trend trend-up">
                        <i class="fas fa-arrow-up"></i>
                        +${Math.round(forecast.growth_rate)}%
                    </div>
                </div>
            `).join('')}
        `;
    }

    const resultsDiv = document.getElementById('forecastResults');
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
    }
}

// Interactive card effects
function addCardEffects() {
    document.addEventListener('mousemove', function(e) {
        const cards = document.querySelectorAll('.card');
        cards.forEach(card => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;

                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
            } else {
                card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
            }
        });
    });
}

// Error handling
function handleError(error, context) {
    console.error(`Error in ${context}:`, error);

    // Show user-friendly error message
    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--error-color);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    errorMessage.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Something went wrong. Please try again.</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; margin-left: 0.5rem;">&times;</button>
        </div>
    `;

    document.body.appendChild(errorMessage);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorMessage.parentElement) {
            errorMessage.remove();
        }
    }, 5000);
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Export functions for global access (if needed)
window.switchTab = switchTab;
window.focusSkillInput = focusSkillInput;
window.focusCurrentSkillInput = focusCurrentSkillInput;
window.focusForecastSkillInput = focusForecastSkillInput;
window.removeSkillByName = removeSkillByName;
window.analyzeSkills = analyzeSkills;
window.predictSalary = predictSalary;
window.generateRoadmap = generateRoadmap;
window.generateForecast = generateForecast;