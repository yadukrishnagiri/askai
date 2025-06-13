# AskAI - AI-Powered CV Analyzer and Interactive Interviewer

An intelligent recruitment tool that leverages Google's Gemini AI to analyze CVs and conduct adaptive technical interviews with real-time feedback and scoring.

## 🚀 Features

### CV Analysis
- **PDF Text Extraction**: Automatically extracts text from PDF resumes
- **AI-Powered Analysis**: Uses Google Gemini AI to analyze technical skills, experience, and achievements
- **Skill Assessment**: Rates technical skills on a 1-5 scale
- **Improvement Suggestions**: Identifies areas for professional development
- **Question Generation**: Creates relevant interview questions based on CV content

### Interactive Interviewing
- **Adaptive Questioning**: Dynamic question generation based on candidate responses
- **Real-time Evaluation**: Instant scoring on technical accuracy and communication
- **Difficulty Adjustment**: Automatically adjusts question difficulty based on performance
- **Project-Based Questions**: Focuses on practical experience and problem-solving
- **Comprehensive Feedback**: Detailed evaluation with improvement suggestions

### Results Management
- **JSON Export**: Saves detailed interview results and analysis
- **Summary Reports**: Generates concise interview summaries
- **Historical Data**: Maintains records of all interviews conducted
- **Structured Scoring**: Consistent evaluation metrics across all interviews

## 📋 Requirements

- Python 3.7+
- Google Gemini API Key
- PDF files for CV analysis

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yadukrishnagiri/askai.git
   cd askai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Get Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate a new API key
   - Add it to your `.env` file

## 🎯 Usage

### Basic Usage

1. **Prepare your CV**: Place your PDF resume in the `data/` directory

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Follow the prompts**:
   - Enter the position you're applying for
   - The system will analyze your CV
   - Participate in the interactive interview
   - Review your results

### CV Analysis Output

The system provides:
- **Technical Skills**: Rated 1-5 stars (★★★★★)
- **Years of Experience**: Extracted from CV content
- **Key Achievements**: Highlighted accomplishments
- **Improvement Areas**: Suggestions for skill development
- **Interview Questions**: Tailored questions for your profile

### Interview Process

- **Adaptive Difficulty**: Questions start basic and adjust based on performance
- **Real-time Feedback**: Immediate scoring and suggestions after each answer
- **Project Focus**: Emphasis on practical experience and problem-solving
- **Professional Evaluation**: Technical accuracy and communication assessment

## 📁 Project Structure

```
askai/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── .env                # Environment variables (create this)
├── data/               # Directory for CV files
│   └── *.pdf          # Place your CV files here
└── Result/            # Interview results and summaries
    ├── interview_*.json
    └── interview_summary.json
```

## 🔧 Configuration

### CV File Path
Update the PDF path in `main.py` (line ~420):
```python
pdf_path = r"path/to/your/cv.pdf"
```

### Results Directory
Modify the results directory in `_save_results()` method:
```python
results_dir = r"your/preferred/results/directory"
```

### Interview Settings
Adjust interview parameters in `InteractiveInterviewer` class:
- `max_questions`: Maximum number of questions (default: 10)
- `timeout`: API timeout in seconds (default: 300)
- `difficulty_levels`: Question difficulty progression

## 📊 Sample Output

### CV Analysis
```
CV Analysis Results
============================================================

Technical Skills:
  • Python               ★★★★☆
  • Machine Learning      ★★★☆☆
  • Data Analysis        ★★★★★
  • SQL                   ★★★☆☆

Years of Experience: 3

Key Achievements:
  • Developed ML model improving accuracy by 25%
  • Led team of 5 developers on data pipeline project
  • Published research paper on NLP applications

Areas for Improvement:
  • Advanced machine learning algorithms
  • Cloud deployment experience
  • System design knowledge
```

### Interview Feedback
```
Question 1 (basic):
Could you describe your most technically challenging project?

Feedback:
Technical Accuracy: 8/10
Communication: 7/10
Feedback: Good explanation of technical challenges. Consider providing more details about specific implementation decisions.
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yadukrishnagiri/askai/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Support for multiple CV formats (Word, text files)
- [ ] Integration with video interview capabilities
- [ ] Advanced analytics and reporting dashboard
- [ ] Multi-language support
- [ ] Integration with ATS systems
- [ ] Batch processing for multiple candidates

---

**Note**: This tool is designed to assist in the recruitment process and should be used as a supplementary tool alongside human judgment in hiring decisions.