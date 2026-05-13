import { useState } from 'react'
import './index.css'
import dialogLogo from './assets/Dialog_Axiata_logo.png'

function App() {
  const [formData, setFormData] = useState({
    district: "Colombo",
    age_group: "26-35",
    current_package: "20GB",
    monthly_data_usage_gb: 35.0,
    youtube_usage_gb: 10.0,
    social_usage_gb: 8.0,
    voice_minutes: 500,
    sms_usage: 100,
    monthly_spend_lkr: 5000,
    reload_frequency: 10,
    add_on_count: 2,
    device_type: "4G_Mobile",
    is_5g_supported: 0,
    sim_type: "4G",
    router_owned: 0,
    churn_risk_score: 0.3,
    complaint_count: 1,
    previous_offer_response: "Ignored",
    days_since_last_change: 180
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    // Keep values as strings in state so dropdowns select properly
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Parse numerical values before sending to API
    const payload = { ...formData };
    
    ['monthly_data_usage_gb', 'youtube_usage_gb', 'social_usage_gb', 'churn_risk_score'].forEach(key => {
      payload[key] = parseFloat(payload[key]);
    });
    
    ['voice_minutes', 'sms_usage', 'monthly_spend_lkr', 'reload_frequency', 'add_on_count', 'is_5g_supported', 'router_owned', 'complaint_count', 'days_since_last_change'].forEach(key => {
      payload[key] = parseInt(payload[key], 10);
    });

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error('Prediction request failed. Please check if the server is running.');
      }

      const data = await response.json();
      setResult(data.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <img src={dialogLogo} alt="Dialog Axiata" className="logo" />
        <h1 className="title">Smart NBA & NBO Predictor</h1>
        <p className="subtitle">AI-powered Next Best Action and Offer recommendation engine for hyper-personalized customer engagement.</p>
      </header>

      <main className="main-content">
        <form className="glass-panel" onSubmit={handleSubmit}>
          <h2 className="section-title">Customer Profile</h2>
          
          <div className="form-grid">
            <div className="input-group">
              <label>District</label>
              <select className="custom-select" name="district" value={formData.district} onChange={handleChange}>
                <option value="Colombo">Colombo</option>
                <option value="Kandy">Kandy</option>
                <option value="Galle">Galle</option>
                <option value="Gampaha">Gampaha</option>
                <option value="Matara">Matara</option>
                <option value="Jaffna">Jaffna</option>
                <option value="Kurunegala">Kurunegala</option>
              </select>
            </div>

            <div className="input-group">
              <label>Age Group</label>
              <select className="custom-select" name="age_group" value={formData.age_group} onChange={handleChange}>
                <option value="18-25">18-25</option>
                <option value="26-35">26-35</option>
                <option value="36-45">36-45</option>
                <option value="46-60">46-60</option>
                <option value="60+">60+</option>
              </select>
            </div>

            <div className="input-group">
              <label>Current Package</label>
              <select className="custom-select" name="current_package" value={formData.current_package} onChange={handleChange}>
                <option value="10GB">10GB</option>
                <option value="20GB">20GB</option>
                <option value="50GB">50GB</option>
                <option value="100GB">100GB</option>
              </select>
            </div>

            <div className="input-group">
              <label>Monthly Data Usage (GB)</label>
              <select className="custom-select" name="monthly_data_usage_gb" value={formData.monthly_data_usage_gb} onChange={handleChange}>
                <option value="5.0">Low (&lt;10 GB)</option>
                <option value="20.0">Medium (10-30 GB)</option>
                <option value="35.0">High (30-50 GB)</option>
                <option value="75.0">Very High (50-100 GB)</option>
                <option value="120.0">Extreme (100+ GB)</option>
              </select>
            </div>

            <div className="input-group">
              <label>YouTube Usage (GB)</label>
              <select className="custom-select" name="youtube_usage_gb" value={formData.youtube_usage_gb} onChange={handleChange}>
                <option value="2.0">Light (2 GB)</option>
                <option value="10.0">Moderate (10 GB)</option>
                <option value="25.0">Heavy (25 GB)</option>
                <option value="50.0">Extreme (50 GB)</option>
              </select>
            </div>

            <div className="input-group">
              <label>Social Media Usage (GB)</label>
              <select className="custom-select" name="social_usage_gb" value={formData.social_usage_gb} onChange={handleChange}>
                <option value="1.0">Light (1 GB)</option>
                <option value="8.0">Moderate (8 GB)</option>
                <option value="20.0">Heavy (20 GB)</option>
              </select>
            </div>

            <div className="input-group">
              <label>Voice Minutes</label>
              <select className="custom-select" name="voice_minutes" value={formData.voice_minutes} onChange={handleChange}>
                <option value="100">100 mins</option>
                <option value="300">300 mins</option>
                <option value="500">500 mins</option>
                <option value="1000">1000+ mins</option>
              </select>
            </div>

            <div className="input-group">
              <label>SMS Count</label>
              <select className="custom-select" name="sms_usage" value={formData.sms_usage} onChange={handleChange}>
                <option value="10">10 SMS</option>
                <option value="100">100 SMS</option>
                <option value="500">500 SMS</option>
              </select>
            </div>

            <div className="input-group">
              <label>Monthly Spend (LKR)</label>
              <select className="custom-select" name="monthly_spend_lkr" value={formData.monthly_spend_lkr} onChange={handleChange}>
                <option value="1000">1,000 LKR</option>
                <option value="3000">3,000 LKR</option>
                <option value="5000">5,000 LKR</option>
                <option value="10000">10,000 LKR</option>
              </select>
            </div>

            <div className="input-group">
              <label>Reload Frequency (per month)</label>
              <select className="custom-select" name="reload_frequency" value={formData.reload_frequency} onChange={handleChange}>
                <option value="2">2 times</option>
                <option value="5">5 times</option>
                <option value="10">10 times</option>
                <option value="20">20+ times</option>
              </select>
            </div>

            <div className="input-group">
              <label>Active Add-ons</label>
              <select className="custom-select" name="add_on_count" value={formData.add_on_count} onChange={handleChange}>
                <option value="0">None (0)</option>
                <option value="1">1 Add-on</option>
                <option value="2">2 Add-ons</option>
                <option value="5">5+ Add-ons</option>
              </select>
            </div>

            <div className="input-group">
              <label>Device Type</label>
              <select className="custom-select" name="device_type" value={formData.device_type} onChange={handleChange}>
                <option value="Basic_Phone">Basic Phone</option>
                <option value="4G_Mobile">4G Mobile</option>
                <option value="5G_Mobile">5G Mobile</option>
              </select>
            </div>

            <div className="input-group">
              <label>5G Supported Device</label>
              <select className="custom-select" name="is_5g_supported" value={formData.is_5g_supported} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>

            <div className="input-group">
              <label>SIM Type</label>
              <select className="custom-select" name="sim_type" value={formData.sim_type} onChange={handleChange}>
                <option value="4G">4G</option>
                <option value="5G">5G</option>
              </select>
            </div>

            <div className="input-group">
              <label>Owns Dialog Router</label>
              <select className="custom-select" name="router_owned" value={formData.router_owned} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>

            <div className="input-group">
              <label>Churn Risk Score</label>
              <select className="custom-select" name="churn_risk_score" value={formData.churn_risk_score} onChange={handleChange}>
                <option value="0.1">Low (0.1)</option>
                <option value="0.3">Moderate (0.3)</option>
                <option value="0.7">High (0.7)</option>
                <option value="0.9">Critical (0.9)</option>
              </select>
            </div>

            <div className="input-group">
              <label>Recent Complaints</label>
              <select className="custom-select" name="complaint_count" value={formData.complaint_count} onChange={handleChange}>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="3">3</option>
                <option value="5">5+</option>
              </select>
            </div>

            <div className="input-group">
              <label>Prev Offer Response</label>
              <select className="custom-select" name="previous_offer_response" value={formData.previous_offer_response} onChange={handleChange}>
                <option value="Accepted">Accepted</option>
                <option value="Ignored">Ignored</option>
                <option value="Rejected">Rejected</option>
              </select>
            </div>

            <div className="input-group">
              <label>Days Since Plan Change</label>
              <select className="custom-select" name="days_since_last_change" value={formData.days_since_last_change} onChange={handleChange}>
                <option value="30">1 Month</option>
                <option value="90">3 Months</option>
                <option value="180">6 Months</option>
                <option value="365">1 Year+</option>
              </select>
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? <div className="loader"></div> : 'Generate Insights'}
          </button>
        </form>

        <div className="glass-panel results-panel">
          <h2 className="section-title">AI Recommendation</h2>
          
          {error && (
            <div style={{ color: '#ff4d63', background: 'rgba(255, 77, 99, 0.1)', padding: '1rem', borderRadius: '8px' }}>
              ⚠️ {error}
            </div>
          )}

          {!result && !error && !loading && (
            <div className="empty-state">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 16v-4"></path>
                <path d="M12 8h.01"></path>
              </svg>
              <p>Configure the customer profile and hit generate to see the next best action and offer.</p>
            </div>
          )}

          {result && (
            <>
              <div className="result-card">
                <div className="result-label">Next Best Action</div>
                <div className="result-value">{(result.predicted_best_action || result.predicted_action || "").replace(/_/g, ' ')}</div>
                
                {result['action_confidence_%'] && (
                  <div style={{ display: 'inline-block', marginTop: '0.5rem', padding: '0.3rem 0.8rem', borderRadius: '20px', background: 'rgba(247, 13, 41, 0.1)', color: '#ff4d63', border: '1px solid rgba(247, 13, 41, 0.3)', fontSize: '0.85rem', fontWeight: '600' }}>
                    {result['action_confidence_%']}% Confidence
                  </div>
                )}

                <div className="result-desc" style={{ marginTop: '1rem' }}>Primary strategy recommended for engagement.</div>
                
                {result.top3_actions && result.top3_actions.length > 1 && (
                  <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                    <div style={{ fontSize: '0.85rem', color: '#a0a0a0', marginBottom: '0.5rem' }}>Top Alternatives:</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {result.top3_actions.slice(1).map((item, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', background: 'rgba(0,0,0,0.3)', padding: '0.6rem 1rem', borderRadius: '8px' }}>
                          <span style={{ color: '#ddd' }}>{item.action.replace(/_/g, ' ')}</span>
                          <span style={{ color: '#ff4d63', fontWeight: '500' }}>{item.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="result-card" style={{ '--primary-color': '#00d2ff' }}>
                <div className="result-label">Next Best Offer</div>
                <div className="result-value" style={{ color: '#00d2ff' }}>{(result.predicted_offer || "").replace(/_/g, ' ')}</div>
                
                {result['offer_confidence_%'] && (
                  <div style={{ display: 'inline-block', marginTop: '0.5rem', padding: '0.3rem 0.8rem', borderRadius: '20px', background: 'rgba(0, 210, 255, 0.1)', color: '#00d2ff', border: '1px solid rgba(0, 210, 255, 0.3)', fontSize: '0.85rem', fontWeight: '600' }}>
                    {result['offer_confidence_%']}% Confidence
                  </div>
                )}

                <div className="result-desc" style={{ marginTop: '1rem' }}>Specific package/offer most likely to be accepted.</div>
                
                {result.top3_offers && result.top3_offers.length > 1 && (
                  <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                    <div style={{ fontSize: '0.85rem', color: '#a0a0a0', marginBottom: '0.5rem' }}>Top Alternatives:</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {result.top3_offers.slice(1).map((item, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', background: 'rgba(0,0,0,0.3)', padding: '0.6rem 1rem', borderRadius: '8px' }}>
                          <span style={{ color: '#ddd' }}>{item.offer.replace(/_/g, ' ')}</span>
                          <span style={{ color: '#00d2ff', fontWeight: '500' }}>{item.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
