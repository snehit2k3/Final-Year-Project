import React, { useState, useRef, useCallback } from 'react';
import { FileText, Zap, Aperture, CheckCircle, Upload, XCircle, Code, Loader2, Database } from 'lucide-react';

// Utility component for consistent Card styling
const Card = ({ title, content, color, icon: Icon, defaultText }) => (
  <div className={`p-4 ${color} border rounded-xl shadow-sm space-y-2`}>
    <div className="flex items-center space-x-2">
      <Icon className={`w-5 h-5 ${color.includes('red') ? 'text-red-700' : color.includes('yellow') ? 'text-yellow-700' : 'text-green-700'}`} />
      <h3 className={`font-semibold ${color.includes('red') ? 'text-red-700' : color.includes('yellow') ? 'text-yellow-700' : 'text-green-700'}`}>
        {title}
      </h3>
    </div>
    <div className="text-sm text-gray-700 whitespace-pre-wrap break-words">
      {content || defaultText}
    </div>
  </div>
);

// Main Application Component
export default function App() {
  const [contractCode, setContractCode] = useState('');
  const [fileName, setFileName] = useState('No contract selected');
  const [selectedModel, setSelectedModel] = useState('rnn');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState({
    vulnerabilities: 'N/A',
    critical: 'Select a contract to view vulnerabilities.',
    optimization: 'Select a contract to view suggestions.',
    compliance: 'Select a contract to view compliance.',
    error: null,
  });

  const fileInputRef = useRef(null);

  const handleFileChange = useCallback((event) => {
    const file = event.target.files[0];
    event.target.value = null; 

    if (file) {
      if (!file.name.toLowerCase().endsWith('.sol')) {
        setResults(prev => ({ 
          ...prev, 
          error: 'Invalid file type. Please select a Solidity (.sol) file.',
          vulnerabilities: 'N/A' 
        }));
        setFileName('Error reading file');
        return;
      }

      setResults(prev => ({ ...prev, error: null }));
      setFileName(`Loading: ${file.name}...`);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const code = e.target.result;
        setContractCode(code);
        setFileName(file.name);
        setResults({
          vulnerabilities: 'N/A',
          critical: 'Contract loaded. Click "Analyze Contract" to begin.',
          optimization: 'Select a contract to view suggestions.',
          compliance: 'Select a contract to view compliance.',
          error: null,
        });
      };
      reader.onerror = () => {
        setResults(prev => ({ ...prev, error: 'Error reading file content.' }));
        setFileName('Error reading file');
      };
      reader.readAsText(file);
    }
  }, []);

  const handleChooseFileClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleAnalyzeContract = async () => {
    if (!contractCode.trim()) {
      setResults(prev => ({ ...prev, error: 'Please paste or load a contract before analyzing.' }));
      return;
    }

    setIsAnalyzing(true);
    setResults(prev => ({ ...prev, error: null }));
    
    // --- UPDATED API LOGIC FOR VITE ---
    let apiUrl;
    
    // 1. Get URLs from Vercel Environment Variables (Must use import.meta.env for Vite)
    const rnnEnvUrl = import.meta.env.VITE_RNN_API;
    const gnnEnvUrl = import.meta.env.VITE_GNN_API;

    // 2. Fallbacks for local testing (localhost)
    // IMPORTANT: On Vercel, these must be defined in Settings > Environment Variables
    const rnnFallback = 'http://127.0.0.1:5001'; 
    const gnnFallback = 'http://127.0.0.1:5002';

    if (selectedModel === 'rnn') {
      const baseUrl = rnnEnvUrl || rnnFallback;
      // Ensure we don't have double slashes if the env var ends in /
      const cleanBase = baseUrl.replace(/\/$/, '');
      apiUrl = `${cleanBase}/predict`; 
    } else { // 'gnn'
      const baseUrl = gnnEnvUrl || gnnFallback;
      const cleanBase = baseUrl.replace(/\/$/, '');
      apiUrl = `${cleanBase}/predict`;
    }
    
    console.log("Attempting to connect to:", apiUrl); 
    // ----------------------------------

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_code: contractCode }),
      });

      const analysisData = await response.json();

      if (!response.ok) {
        throw new Error(analysisData.error || `API request failed: ${response.status}`);
      }
      
      let criticalText, complianceText, vulnCount;
      
      // Use the JSON response fields based on the selected model
      if (analysisData.vulnerable) {
        vulnCount = 1; // Both servers use 'vulnerable: true'
        
        if (selectedModel === 'rnn') {
          // RNN-specific fields
          criticalText = `STATUS: ${analysisData.status} (Confidence: ${analysisData.confidence_rnn})\n\n` +
                         `MODEL: ${analysisData.model_type}\n\n` +
                         `RULE TRIGGERED: ${analysisData.rule_triggered}\n\n` +
                         `PROBLEM: ${analysisData.problem}\n\n` +
                         `RISK: ${analysisData.risk}\n\n` +
                         `FIX: ${analysisData.fix}`;
        } else {
          // GNN-specific fields
          criticalText = `STATUS: ${analysisData.status} (Confidence: ${analysisData.confidence})\n\n` +
                         `VULNERABILITY: ${analysisData.vulnerability}\n\n` +
                         `PROBLEM: ${analysisData.problem}\n\n` +
                         `RISK: ${analysisData.risk}\n\n` +
                         `FIX: ${analysisData.fix}`;
        }
        complianceText = 'STATUS: NON-COMPLIANT. Critical vulnerabilities prevent standard compliance score.';
      } else {
        vulnCount = 0;

        if (selectedModel === 'rnn') {
          // RNN-specific fields
          criticalText = `STATUS: ${analysisData.status} (Confidence: ${analysisData.confidence_rnn})\n\n` +
                         `MODEL: ${analysisData.model_type}\n\n` +
                         `RULE TRIGGERED: ${analysisData.rule_triggered}\n\n` +
                         `${analysisData.summary}\n\n` + 
                         `Always verify with a manual audit.`;
        } else {
          // GNN-specific fields
          criticalText = `STATUS: ${analysisData.status} (Confidence: ${analysisData.confidence})\n\n` +
                         `${analysisData.summary}\n\n` + 
                         `Always verify with a manual audit.`;
        }
        complianceText = 'STATUS: FULLY COMPLIANT. Meets all baseline security and best practice requirements.';
      }

      setResults({
        vulnerabilities: vulnCount,
        critical: criticalText,
        optimization: 'Optimization analysis is not yet available with this model.', 
        compliance: complianceText,
        error: null,
      });

    } catch (error) {
      console.error("API call error:", error);
      setResults(prev => ({
        ...prev,
        error: error.message || 'An unknown error occurred. Could not connect to the backend server.',
        vulnerabilities: 'N/A',
      }));
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const vulnCount = typeof results.vulnerabilities === 'number' ? results.vulnerabilities : 0;
  const vulnsTextColor = vulnCount > 0 ? 'text-red-500' : 'text-green-500';
  
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-2 sm:p-4 font-sans">
      <div className="w-full max-w-7xl bg-white shadow-2xl border border-gray-100 rounded-3xl p-4 md:p-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <div className="lg:col-span-2 flex flex-col space-y-6">
          <h2 className="text-2xl lg:text-3xl font-extrabold text-gray-900 flex items-center">
            <Code className="w-6 h-6 lg:w-7 lg:h-7 mr-3 text-blue-600" />
            Smart Contract Analysis
          </h2>

          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".sol" className="hidden" />
          
          <textarea
            placeholder="select your smart contract or paste contract code (Solidity)..."
            value={contractCode}
            onChange={(e) => {
              setContractCode(e.target.value);
              setResults(prev => ({
                ...prev, vulnerabilities: 'N/A',
                critical: 'Code modified. Re-analyze to update results.',
                optimization: 'Code modified. Re-analyze to update results.',
                compliance: 'Code modified. Re-analyze to update results.',
                error: null,
              }));
            }}
            className="w-full h-64 sm:h-80 lg:h-[450px] border border-gray-300 rounded-xl p-4 font-mono text-sm bg-gray-50 focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition duration-150 shadow-inner"
          />

          {results.error && (
            <div className="flex items-center p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
              <XCircle className="w-4 h-4 mr-2 flex-shrink-0" />
              <span>{results.error}</span>
            </div>
          )}

          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
            <div className="relative">
              <Database className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isAnalyzing}
                className="appearance-none w-full sm:w-auto flex items-center justify-center pl-10 pr-6 py-3 bg-gray-200 text-gray-800 rounded-xl font-medium hover:bg-gray-300 transition duration-150 disabled:opacity-60 disabled:cursor-not-allowed shadow-md focus:outline-none focus:ring-2 focus:ring-gray-400"
              >
                <option value="rnn">RNN Model</option>
                <option value="gnn">GNN Model</option>
              </select>
            </div>

            <button 
              onClick={handleChooseFileClick}
              disabled={isAnalyzing}
              className="flex items-center justify-center px-6 py-3 bg-gray-200 text-gray-800 rounded-xl font-medium hover:bg-gray-300 transition duration-150 disabled:opacity-60 disabled:cursor-not-allowed shadow-md"
            >
              <Upload className="w-5 h-5 mr-2" />
              Choose .sol File
            </button>
            <button 
              onClick={handleAnalyzeContract}
              disabled={isAnalyzing || !contractCode.trim()}
              className="flex items-center justify-center px-6 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition duration-150 disabled:bg-blue-300 shadow-lg shadow-blue-200"
            >
              {isAnalyzing ? (
                <><Loader2 className="w-5 h-5 mr-2 animate-spin" />Analyzing...</>
              ) : (
                <><Zap className="w-5 h-5 mr-2" />Analyze Contract</>
              )}
            </button>
          </div>

          <div className="p-4 bg-white border border-gray-200 rounded-xl shadow-md mt-4">
            <h3 className="text-lg lg:text-xl font-semibold text-gray-800 mb-2 border-b pb-2">
              <FileText className="w-5 h-5 inline mr-2 text-gray-500" />
              Contract Status
            </h3>
            <p className="text-sm text-gray-600 flex flex-wrap items-center">
              <span className="font-mono text-gray-700 font-semibold mr-2 break-all">{fileName}</span>
              <span className={`font-bold text-nowrap ${vulnsTextColor}`}>
                {isAnalyzing ? (
                  <span className="text-blue-500">Analysis in progress...</span>
                ) : (
                  <>• {typeof results.vulnerabilities === 'number' ? `${results.vulnerabilities} vulnerability${results.vulnerabilities !== 1 ? 's' : ''}` : results.vulnerabilities}</>
                )}
              </span>
            </p>
          </div>
        </div>

        <div className="flex flex-col space-y-6">
          <h2 className="text-2xl lg:text-3xl font-extrabold text-gray-900 flex items-center">
            <Aperture className="w-6 h-6 lg:w-7 lg:h-7 mr-3 text-green-600" />
            Security Insights
          </h2>
          <Card
            title="Critical Vulnerabilities"
            content={results.critical}
            color="bg-red-50 border-red-300"
            icon={Zap}
            defaultText="Select a contract and analyze to view vulnerabilities."
          />
          <Card
            title="Optimization Suggestions"
            content={results.optimization}
            color="bg-yellow-50 border-yellow-300"
            icon={Aperture}
            defaultText="Select a contract and analyze to view suggestions."
          />
          <Card
            title="Compliance Status"
            content={results.compliance}
            color="bg-green-50 border-green-300"
            icon={CheckCircle}
            defaultText="Select a contract and analyze to view compliance."
          />
        </div>
      </div>
    </div>
  );
}