import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';

const Assessment = ({ user, mlModel }) => {
  const { skill } = useParams();
  const navigate = useNavigate();
  
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [timeLeft, setTimeLeft] = useState(300); // 5 minutes
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [completed, setCompleted] = useState(false);
  const [results, setResults] = useState(null);

  useEffect(() => {
    loadAssessment();
  }, [skill]);

  useEffect(() => {
    if (timeLeft > 0 && !completed) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    } else if (timeLeft === 0) {
      handleSubmitAssessment();
    }
  }, [timeLeft, completed]);

  const loadAssessment = async () => {
    try {
      setLoading(true);
      
      // Generate questions based on skill
      const generatedQuestions = generateQuestions(skill);
      setQuestions(generatedQuestions);
      
    } catch (error) {
      console.error('Error loading assessment:', error);
      toast.error('Failed to load assessment');
    } finally {
      setLoading(false);
    }
  };

  const generateQuestions = (skillName) => {
    const questionBank = {
      python: [
        {
          question: "What is the output of print(type([]) == list)?",
          options: ["True", "False", "Error", "None"],
          correct: 0,
          difficulty: "beginner"
        },
        {
          question: "Which method is used to add an element to a list in Python?",
          options: ["add()", "append()", "insert()", "push()"],
          correct: 1,
          difficulty: "beginner"
        },
        {
          question: "What is a lambda function in Python?",
          options: [
            "A named function",
            "An anonymous function",
            "A class method",
            "A built-in function"
          ],
          correct: 1,
          difficulty: "intermediate"
        },
        {
          question: "Which decorator is used to create a property in Python?",
          options: ["@property", "@staticmethod", "@classmethod", "@decorator"],
          correct: 0,
          difficulty: "intermediate"
        },
        {
          question: "What is the Global Interpreter Lock (GIL) in Python?",
          options: [
            "A security feature",
            "A memory management tool",
            "A mechanism that prevents multiple threads from executing Python code simultaneously",
            "A debugging tool"
          ],
          correct: 2,
          difficulty: "advanced"
        }
      ],
      javascript: [
        {
          question: "What is the difference between '==' and '===' in JavaScript?",
          options: [
            "No difference",
            "'==' checks type and value, '===' checks only value",
            "'===' checks type and value, '==' checks only value",
            "'==' is for assignment, '===' is for comparison"
          ],
          correct: 2,
          difficulty: "beginner"
        },
        {
          question: "What is a closure in JavaScript?",
          options: [
            "A way to close a function",
            "A function that has access to variables in its outer scope",
            "A method to end a loop",
            "A type of error handling"
          ],
          correct: 1,
          difficulty: "intermediate"
        }
      ],
      "machine learning": [
        {
          question: "What is overfitting in machine learning?",
          options: [
            "Model performs well on training but poor on test data",
            "Model performs poorly on both training and test data",
            "Model is too simple",
            "Model has too few parameters"
          ],
          correct: 0,
          difficulty: "intermediate"
        },
        {
          question: "What is the purpose of cross-validation?",
          options: [
            "To increase training speed",
            "To reduce model complexity",
            "To assess model performance and generalization",
            "To clean the data"
          ],
          correct: 2,
          difficulty: "intermediate"
        }
      ]
    };

    const skillQuestions = questionBank[skillName.toLowerCase()] || questionBank.python;
    return skillQuestions.slice(0, 10); // Limit to 10 questions
  };

  const handleAnswerSelect = (answerIndex) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = {
      questionIndex: currentQuestion,
      selectedAnswer: answerIndex,
      correct: answerIndex === questions[currentQuestion].correct,
      timeSpent: 300 - timeLeft
    };
    setAnswers(newAnswers);
  };

  const handleNextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      handleSubmitAssessment();
    }
  };

  const handleSubmitAssessment = async () => {
    try {
      setLoading(true);
      
      // Use ML model to assess skill level
      const assessmentResult = await mlModel.assessSkill(skill, answers);
      setResults(assessmentResult);
      setCompleted(true);
      
      toast.success('Assessment completed!');
    } catch (error) {
      console.error('Error submitting assessment:', error);
      toast.error('Failed to submit assessment');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <div className="assessment-loading">
        <div className="loading-spinner"></div>
        <p>Loading {skill} assessment...</p>
      </div>
    );
  }

  if (completed && results) {
    return (
      <div className="assessment-results">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="results-container"
        >
          <div className="results-header">
            <CheckCircleIcon className="w-16 h-16 text-green-500" />
            <h1>Assessment Complete!</h1>
            <p>Here are your results for {skill}</p>
          </div>

          <div className="results-stats">
            <div className="stat-item">
              <h3>Skill Level</h3>
              <div className="level-indicator">
                <div 
                  className="level-bar"
                  style={{ width: `${results.estimatedLevel * 100}%` }}
                ></div>
              </div>
              <p>{(results.estimatedLevel * 100).toFixed(0)}%</p>
            </div>

            <div className="stat-item">
              <h3>Confidence</h3>
              <p>{(results.confidence * 100).toFixed(0)}%</p>
            </div>

            <div className="stat-item">
              <h3>Correct Answers</h3>
              <p>{answers.filter(a => a.correct).length} / {questions.length}</p>
            </div>
          </div>

          <div className="recommendations">
            <h3>Recommendations</h3>
            <ul>
              {results.recommendations.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>

          <div className="results-actions">
            <button
              onClick={() => navigate('/dashboard')}
              className="btn-primary"
            >
              Back to Dashboard
            </button>
            <button
              onClick={() => navigate('/learning-path')}
              className="btn-secondary"
            >
              View Learning Path
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  const currentQ = questions[currentQuestion];
  const progress = ((currentQuestion + 1) / questions.length) * 100;

  return (
    <div className="assessment">
      <div className="assessment-header">
        <div className="assessment-info">
          <h1>{skill} Assessment</h1>
          <p>Question {currentQuestion + 1} of {questions.length}</p>
        </div>
        
        <div className="assessment-timer">
          <ClockIcon className="w-5 h-5" />
          <span className={timeLeft < 60 ? 'text-red-500' : ''}>
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>

      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ width: `${progress}%` }}
        ></div>
      </div>

      <motion.div
        key={currentQuestion}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="question-container"
      >
        <div className="question">
          <h2>{currentQ.question}</h2>
          <div className="difficulty-badge">
            {currentQ.difficulty}
          </div>
        </div>

        <div className="options">
          {currentQ.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswerSelect(index)}
              className={`option ${
                answers[currentQuestion]?.selectedAnswer === index ? 'selected' : ''
              }`}
            >
              <span className="option-letter">
                {String.fromCharCode(65 + index)}
              </span>
              <span className="option-text">{option}</span>
            </button>
          ))}
        </div>

        <div className="question-actions">
          <button
            onClick={handleNextQuestion}
            disabled={!answers[currentQuestion]}
            className="btn-primary"
          >
            {currentQuestion === questions.length - 1 ? 'Finish Assessment' : 'Next Question'}
            <ArrowRightIcon className="w-4 h-4 ml-2" />
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default Assessment;
