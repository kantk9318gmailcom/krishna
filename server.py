import os
import csv
import json
from flask import Flask, render_template_string, jsonify, request
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_DIR', 'src')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- AI Agent Setup ---
# You must replace the placeholder with your actual API key.
# It is hardcoded here to avoid deployment issues with environment variables.
llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-2.5-flash",
    temperature=0.1
)

def read_student_data():
    """Reads student data from a CSV file and returns it as a formatted string."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'students.csv')
    students = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Use .strip() to handle potential whitespace in column names and values
                students.append(
                    f"Student {row['name'].strip()}: Course = {row['course'].strip()}, Academic Performance = {row['academic_performance'].strip()}, Schedule = {row['schedule'].strip()}"
                )
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return "\n".join(students)

def run_matching_agent(student_data):
    """Runs the CrewAI agent to match students."""
    
    if not student_data:
        return "Student data is empty. Please upload a CSV file with data."
    
    matcher_agent = Agent(
        role='Student Matcher',
        goal='Identify and suggest optimal student pairings for collaboration based on academic needs and schedules.',
        backstory='An expert in educational technology and collaborative learning. Your job is to analyze student data and find the best matches for study groups, project collaboration, or peer tutoring.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    matching_task = Task(
        description=f"Analyze the following student data and provide a list of recommended matches. The recommendations should be justified based on course, academic performance, and schedule. Specifically, find peers for study groups (same course, similar schedule), project collaboration (similar skills, complementary schedules), or peer tutoring (one student needs help, another is strong in that area).\n\nStudent Data:\n{student_data}",
        agent=matcher_agent,
        expected_output='A clear, bulleted list of recommended student pairings with a brief explanation for each pairing. Example: "- Student A and Student D: Both are in CS101 and have matching schedules, making them perfect for a study group."'
    )

    crew = Crew(
        agents=[matcher_agent],
        tasks=[matching_task],
        process=Process.sequential,
        verbose=True,
        full_output=True,
    )
    
    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        return f"An error occurred during crew execution: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def home():
    """Serves the main web page with the React app."""
    react_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Student Matching Agent</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
            <div id="root" class="w-full max-w-4xl"></div>
            <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
            <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
            <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

            {% raw %}
            <script type="text/babel">
                const { useState } = React;

                const App = () => {
                    const [results, setResults] = useState('');
                    const [isLoading, setIsLoading] = useState(false);
                    const [error, setError] = useState(null);
                    const [csvFile, setCsvFile] = useState(null);

                    const handleFileChange = (event) => {
                        setCsvFile(event.target.files[0]);
                    };

                    const runAgent = async () => {
                        if (!csvFile) {
                            setError("Please upload a CSV file first.");
                            return;
                        }

                        setIsLoading(true);
                        setError(null);
                        setResults('');

                        const formData = new FormData();
                        formData.append('file', csvFile);

                        try {
                            const uploadResponse = await fetch('/upload-data', {
                                method: 'POST',
                                body: formData,
                            });

                            if (!uploadResponse.ok) {
                                const errorData = await uploadResponse.json();
                                setError(errorData.error || 'Failed to upload file.');
                                setIsLoading(false);
                                return;
                            }

                            const runResponse = await fetch('/run', { method: 'POST' });
                            const data = await runResponse.json();
                            
                            if (runResponse.ok) {
                                setResults(data.output);
                            } else {
                                setError(data.output || 'An unknown error occurred.');
                            }

                        } catch (err) {
                            setError('An error occurred. Please check the server logs.');
                            console.error('Fetch error:', err);
                        } finally {
                            setIsLoading(false);
                        }
                    };

                    return (
                        <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-4xl text-center">
                            <h1 className="text-4xl font-extrabold mb-4 text-green-800">Student Matching Agent</h1>
                            <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
                                This intelligent agent analyzes student data to recommend study groups, peer tutoring, and project collaboration pairings.
                            </p>
                            
                            <div className="flex flex-col items-center justify-center space-y-4">
                                <label className="flex items-center space-x-2 bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-full cursor-pointer transition duration-300 ease-in-out shadow-lg">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                        <path d="M5.5 13a.5.5 0 01.5-.5h2a.5.5 0 010 1H6a.5.5 0 01-.5-.5zM8 10a.5.5 0 00.5.5h2a.5.5 0 000-1h-2A.5.5 0 008 10zM5.5 7a.5.5 0 01.5-.5h2a.5.5 0 010 1H6a.5.5 0 01-.5-.5z" />
                                        <path fillRule="evenodd" d="M11 2a1 1 0 00-1 1v2a1 1 0 002 0V3a1 1 0 00-1-1zM9 9a1 1 0 012 0v5a1 1 0 11-2 0V9zM5.5 13a.5.5 0 01.5-.5h2a.5.5 0 010 1H6a.5.5 0 01-.5-.5zM8 10a.5.5 0 00.5.5h2a.5.5 0 000-1h-2A.5.5 0 008 10zM5.5 7a.5.5 0 01.5-.5h2a.5.5 0 010 1H6a.5.5 0 01-.5-.5z" clipRule="evenodd" />
                                    </svg>
                                    <span>{csvFile ? csvFile.name : 'Choose CSV File'}</span>
                                    <input type="file" onChange={handleFileChange} accept=".csv" className="hidden" />
                                </label>
                                <button
                                    onClick={runAgent}
                                    disabled={isLoading || !csvFile}
                                    className="px-8 py-4 bg-green-700 text-white font-extrabold rounded-full shadow-lg hover:bg-green-800 transition duration-300 ease-in-out disabled:bg-gray-400 disabled:cursor-not-allowed transform hover:scale-105"
                                >
                                    {isLoading ? 'Running Agent...' : 'Run Matching Agent'}
                                </button>
                            </div>
                            
                            <div className="mt-12 text-left bg-gray-50 p-8 rounded-xl shadow-inner border border-gray-200 min-h-64">
                                <h2 className="text-2xl font-bold mb-4 text-green-700" style={{ display: results || error ? 'block' : 'none' }}>Recommendations:</h2>
                                {isLoading && (
                                    <div className="flex flex-col items-center space-y-4">
                                        <div className="w-16 h-16 border-4 border-green-500 border-dotted rounded-full animate-spin"></div>
                                        <p className="text-gray-500 italic">Analyzing student data and generating recommendations...</p>
                                    </div>
                                )}
                                {error && <p className="text-red-500 font-medium whitespace-pre-wrap">{error}</p>}
                                {results && (
                                    <div className="space-y-6">
                                        {results.split('*').map((item, index) => {
                                            if (item.trim() === '') return null;
                                            const parts = item.split(':');
                                            const heading = parts.shift().trim();
                                            const content = parts.join(':').trim();
                                            return (
                                                <div key={index} className="bg-white p-6 rounded-lg shadow-md border-l-4 border-green-500">
                                                    <h3 className="text-lg font-semibold text-green-800">{heading}</h3>
                                                    <p className="text-gray-700 mt-2">{content}</p>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                };

                const root = ReactDOM.createRoot(document.getElementById('root'));
                root.render(<App />);
            </script>
            {% endraw %}
        </body>
        </html>
    """
    return render_template_string(react_html)

# The API endpoint to handle CSV file uploads
@app.route('/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'students.csv')
        try:
            file.save(filepath)
            return jsonify({"message": "File uploaded successfully"}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a .csv file"}), 400

# The API endpoint to run the agent with the latest CSV data
@app.route('/run', methods=['POST'])
def run_agent_endpoint():
    try:
        student_data = read_student_data()
        if not student_data:
            return jsonify({"output": "Student data file is empty or not found."}), 400
        
        result = run_matching_agent(student_data)

        if "An error occurred" in result:
             # Return the error message with a 400 status code
            return jsonify({"output": result}), 400
        else:
            return jsonify({"output": result})
    except Exception as e:
        return jsonify({"output": f"An unhandled error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # You may need to install flask with: pip install Flask
    app.run(debug=True, port=8000)
