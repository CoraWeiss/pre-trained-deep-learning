import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Papa from 'papaparse';

const ImageClassificationDashboard = () => {
  const [data, setData] = useState([]);
  const [topSubjects, setTopSubjects] = useState([]);
  const [confidenceStats, setConfidenceStats] = useState({
    average: 0,
    max: { value: 0, image: '', subject: '' },
    min: { value: 1, image: '', subject: '' }
  });
  const [subjectStats, setSubjectStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('frequency');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Read the file
        const fileContent = await window.fs.readFile('paste.txt', { encoding: 'utf8' });
        
        // Parse the CSV
        const parsedData = Papa.parse(fileContent, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true
        });
        
        setData(parsedData.data);
        processData(parsedData.data);
        setLoading(false);
      } catch (error) {
        console.error("Error loading data:", error);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const processData = (imageData) => {
    // Count subject frequencies
    const subjects = {};
    let totalConfidence = 0;
    let maxConfidence = 0;
    let maxConfidenceImage = '';
    let maxConfidenceSubject = '';
    let minConfidence = 1;
    let minConfidenceImage = '';
    let minConfidenceSubject = '';

    imageData.forEach(row => {
      // Count subjects
      if (subjects[row.subject]) {
        subjects[row.subject]++;
      } else {
        subjects[row.subject] = 1;
      }
      
      // Track confidence statistics
      totalConfidence += row.confidence;
      
      if (row.confidence > maxConfidence) {
        maxConfidence = row.confidence;
        maxConfidenceImage = row.image;
        maxConfidenceSubject = row.subject;
      }
      
      if (row.confidence < minConfidence) {
        minConfidence = row.confidence;
        minConfidenceImage = row.image;
        minConfidenceSubject = row.subject;
      }
    });

    // Sort subjects by frequency and get top 10
    const sortedSubjects = Object.entries(subjects)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([subject, count]) => ({ subject, count }));

    setTopSubjects(sortedSubjects);

    // Set confidence stats
    setConfidenceStats({
      average: totalConfidence / imageData.length,
      max: { 
        value: maxConfidence, 
        image: maxConfidenceImage, 
        subject: maxConfidenceSubject 
      },
      min: { 
        value: minConfidence, 
        image: minConfidenceImage, 
        subject: minConfidenceSubject 
      }
    });

    // Calculate per-subject stats
    const subjectStatsObj = {};

    // Calculate confidence statistics for each subject
    Object.keys(subjects).forEach(subject => {
      const subjectRows = imageData.filter(row => row.subject === subject);
      const confidences = subjectRows.map(row => row.confidence);
      
      const avgConfidence = confidences.reduce((sum, val) => sum + val, 0) / confidences.length;
      const maxConf = Math.max(...confidences);
      const minConf = Math.min(...confidences);
      
      subjectStatsObj[subject] = {
        count: subjectRows.length,
        avgConfidence: avgConfidence,
        maxConfidence: maxConf,
        minConfidence: minConf
      };
    });

    // Get top subjects by average confidence (with at least 5 occurrences)
    const topConfidentSubjects = Object.entries(subjectStatsObj)
      .filter(([_, stats]) => stats.count >= 5)
      .sort((a, b) => b[1].avgConfidence - a[1].avgConfidence)
      .slice(0, 10)
      .map(([subject, stats]) => ({
        subject,
        count: stats.count,
        avgConfidence: stats.avgConfidence,
        maxConfidence: stats.maxConfidence,
        minConfidence: stats.minConfidence
      }));

    setSubjectStats(topConfidentSubjects);
  };

  const formatConfidence = (value) => {
    return (value * 100).toFixed(2) + '%';
  };

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading data...</div>;
  }

  return (
    <div className="p-4 max-w-6xl mx-auto">
      <div className="mb-6 bg-white p-6 rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-4">Image Classification Analysis</h1>
        <p className="mb-2">Total images analyzed: <span className="font-semibold">{data.length}</span></p>
        <p className="mb-2">Unique subjects detected: <span className="font-semibold">{Object.keys(data.reduce((acc, row) => ({...acc, [row.subject]: true}), {})).length}</span></p>
        <p className="mb-2">Average confidence score: <span className="font-semibold">{formatConfidence(confidenceStats.average)}</span></p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold">Highest confidence detection:</h3>
            <p>Subject: <span className="font-bold">{confidenceStats.max.subject}</span></p>
            <p>Image: {confidenceStats.max.image}</p>
            <p>Confidence: <span className="text-green-600 font-bold">{formatConfidence(confidenceStats.max.value)}</span></p>
          </div>
          <div className="bg-red-50 p-4 rounded-lg">
            <h3 className="font-semibold">Lowest confidence detection:</h3>
            <p>Subject: <span className="font-bold">{confidenceStats.min.subject}</span></p>
            <p>Image: {confidenceStats.min.image}</p>
            <p>Confidence: <span className="text-red-600 font-bold">{formatConfidence(confidenceStats.min.value)}</span></p>
          </div>
        </div>
      </div>

      <div className="mb-6 bg-white p-6 rounded-lg shadow">
        <div className="flex mb-4 border-b">
          <button 
            className={`px-4 py-2 ${activeTab === 'frequency' ? 'border-b-2 border-blue-500 text-blue-500 font-bold' : 'text-gray-600'}`}
            onClick={() => setActiveTab('frequency')}
          >
            Subject Frequency
          </button>
          <button 
            className={`px-4 py-2 ${activeTab === 'confidence' ? 'border-b-2 border-blue-500 text-blue-500 font-bold' : 'text-gray-600'}`}
            onClick={() => setActiveTab('confidence')}
          >
            Confidence by Subject
          </button>
        </div>

        {activeTab === 'frequency' && (
          <>
            <h2 className="text-xl font-bold mb-4">Top 10 Subjects by Frequency</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topSubjects}
                  margin={{ top: 10, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="subject" 
                    angle={-45} 
                    textAnchor="end" 
                    height={80} 
                    tick={{ fontSize: 12 }} 
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value} images`, 'Count']} />
                  <Legend />
                  <Bar dataKey="count" fill="#3B82F6" name="Number of Images" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {activeTab === 'confidence' && (
          <>
            <h2 className="text-xl font-bold mb-4">Top 10 Subjects by Average Confidence (min 5 occurrences)</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={subjectStats}
                  margin={{ top: 10, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="subject" 
                    angle={-45} 
                    textAnchor="end" 
                    height={80} 
                    tick={{ fontSize: 12 }} 
                  />
                  <YAxis domain={[0, 1]} tickFormatter={(value) => formatConfidence(value)} />
                  <Tooltip 
                    formatter={(value, name) => [
                      formatConfidence(value), 
                      name === 'avgConfidence' ? 'Average Confidence' : 
                      name === 'maxConfidence' ? 'Max Confidence' :
                      'Min Confidence'
                    ]} 
                  />
                  <Legend />
                  <Bar dataKey="avgConfidence" fill="#10B981" name="Average Confidence" />
                  <Bar dataKey="maxConfidence" fill="#3B82F6" name="Max Confidence" />
                  <Bar dataKey="minConfidence" fill="#EF4444" name="Min Confidence" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>

      <div className="bg-white p-6 rounded-lg shadow overflow-x-auto">
        <h2 className="text-xl font-bold mb-4">Subject Details (Top 10 by Confidence)</h2>
        <table className="min-w-full divide-y divide-gray-200">
          <thead>
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Subject</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Confidence</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max Confidence</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Min Confidence</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {subjectStats.map((subject, index) => (
              <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{subject.subject}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{subject.count}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatConfidence(subject.avgConfidence)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatConfidence(subject.maxConfidence)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatConfidence(subject.minConfidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ImageClassificationDashboard;
