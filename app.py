from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
import time
import threading
from queue import Queue, Empty
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_NAME = "rayanahsan/Llama-2-7b-chat-finetune"

class CardiologyAssistant:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.is_loaded = False
        self.is_busy = False
        self.request_queue = Queue(maxsize=5)  # Handle up to 5 queued requests
        self.response_cache = {}  # Simple caching
        self.stats = {
            'total_requests': 0,
            'avg_response_time': 0,
            'cache_hits': 0
        }
        
    def load_model(self):
        """Load and optimize the model"""
        try:
            logger.info("🔄 Loading Cardiology AI Assistant...")
            
            # Load tokenizer
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimizations
            logger.info("🧠 Loading model (this may take 5-10 minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="cpu",
                torch_dtype=torch.float32,  # More stable on CPU
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Create pipeline
            logger.info("⚙️ Creating inference pipeline...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            self.is_loaded = True
            logger.info("✅ Model loaded successfully! Ready to answer questions.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e
    
    def get_cache_key(self, question):
        """Generate cache key for question"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def get_cached_response(self, question):
        """Check for cached response"""
        cache_key = self.get_cache_key(question)
        if cache_key in self.response_cache:
            self.stats['cache_hits'] += 1
            return self.response_cache[cache_key]
        return None
    
    def cache_response(self, question, answer, response_time):
        """Cache the response"""
        cache_key = self.get_cache_key(question)
        self.response_cache[cache_key] = {
            'answer': answer,
            'cached_at': time.time(),
            'response_time': response_time
        }
        
        # Limit cache size
        if len(self.response_cache) > 100:
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k]['cached_at'])
            del self.response_cache[oldest_key]
    
    def generate_response(self, question):
        """Generate response with optimizations"""
        if not self.is_loaded:
            return "❌ Model is still loading. Please wait a few minutes.", False, 0
        
        # Check cache first
        cached = self.get_cached_response(question)
        if cached:
            logger.info(f"📋 Cache hit for question: {question[:50]}...")
            return f"⚡ {cached['answer']}", True, cached['response_time']
        
        try:
            start_time = time.time()
            self.is_busy = True
            
            # Format question
            prompt = f"<s>[INST] {question} [/INST]"
            
            logger.info(f"🤔 Processing: {question[:50]}...")
            
            # Generate response
            with torch.no_grad():
                output = self.pipe(
                    prompt,
                    max_new_tokens=100,  # Balanced length
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract answer
            if isinstance(output, list) and len(output) > 0:
                answer = output[0].get("generated_text", "").strip()
            else:
                answer = str(output).strip()
            
            answer = answer.replace("</s>", "").strip()
            
            if not answer:
                answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            response_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_requests'] += 1
            if self.stats['avg_response_time'] == 0:
                self.stats['avg_response_time'] = response_time
            else:
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + response_time) 
                    / self.stats['total_requests']
                )
            
            # Cache the response
            self.cache_response(question, answer, response_time)
            
            logger.info(f"✅ Response generated in {response_time:.1f}s")
            return answer, True, response_time
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}", False, 0
        finally:
            self.is_busy = False

# Global assistant instance
assistant = CardiologyAssistant()

@app.route('/')
def home():
    """Serve the main interface"""
    try:
        return render_template('index.html')
    except:
        return """
        <div style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>🫀 Cardiology AI Assistant</h1>
            <h2>Backend is Running!</h2>
            <p>Create <code>templates/index.html</code> with the frontend to use the web interface.</p>
            <p><strong>API Endpoints:</strong></p>
            <ul style="list-style: none; padding: 0;">
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/stats">Statistics</a></li>
                <li>POST /api/ask - Ask questions</li>
            </ul>
            <div style="margin-top: 30px; padding: 20px; background: #f0f8ff; border-radius: 10px;">
                <h3>Model Status</h3>
                <p>Loaded: <strong>{}</strong></p>
                <p>Busy: <strong>{}</strong></p>
            </div>
        </div>
        """.format(
            "✅ Yes" if assistant.is_loaded else "⏳ Loading...",
            "🔄 Yes" if assistant.is_busy else "🟢 Ready"
        )

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Handle question requests"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        if len(question) > 200:
            return jsonify({
                'error': 'Please keep questions under 200 characters for optimal performance'
            }), 400
        
        # Check if model is still loading
        if not assistant.is_loaded:
            return jsonify({
                'error': 'Model is still loading. Please wait a few minutes and try again.',
                'status': 'loading'
            }), 503
        
        # Check if currently processing another request
        if assistant.is_busy:
            return jsonify({
                'error': 'Currently processing another request. Please wait a moment.',
                'status': 'busy',
                'tip': 'Try asking a different question or wait 30-60 seconds'
            }), 429
        
        # Generate response
        answer, success, response_time = assistant.generate_response(question)
        
        if success:
            return jsonify({
                'question': question,
                'answer': answer,
                'response_time': round(response_time, 1),
                'status': 'success',
                'cached': '⚡' in answer  # Indicates if response was cached
            })
        else:
            return jsonify({
                'error': answer,
                'status': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health and status check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': assistant.is_loaded,
        'model_busy': assistant.is_busy,
        'ready_for_requests': assistant.is_loaded and not assistant.is_busy,
        'uptime': time.time()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get usage statistics"""
    return jsonify({
        'total_requests': assistant.stats['total_requests'],
        'average_response_time': round(assistant.stats['avg_response_time'], 1),
        'cache_hits': assistant.stats['cache_hits'],
        'cache_size': len(assistant.response_cache),
        'model_status': {
            'loaded': assistant.is_loaded,
            'busy': assistant.is_busy
        }
    })

#preload some questions to make it faster
@app.route('/api/sample-questions', methods=['GET'])
def sample_questions():
    """Get sample questions users can try"""
    samples = [
        "What are the symptoms of heart failure?",
        "How is hypertension treated?",
        "What is atrial fibrillation?",
        "Explain the difference between systolic and diastolic pressure",
        "What are the risk factors for coronary artery disease?",
        "How is a heart attack diagnosed?",
        "What are ACE inhibitors?",
        "What is an echocardiogram used for?",
        "What are the signs of a stroke?",
        "How does cholesterol affect heart health?"
    ]
    
    return jsonify({
        'questions': samples,
        'tip': 'These questions are optimized for quick responses!'
    })

def start_model_loading():
    """Start model loading in background"""
    def load_in_thread():
        try:
            assistant.load_model()
        except Exception as e:
            logger.error(f"Failed to load model in background: {e}")
    
    thread = threading.Thread(target=load_in_thread)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    print("Starting LLM...")
    
    #start loading model 
    start_model_loading()
    
    #Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )