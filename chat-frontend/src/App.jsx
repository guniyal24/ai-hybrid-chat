import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_URL = "http://127.0.0.1:8000/api";

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [isLoginView, setIsLoginView] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- HELPER: Remove <thinking> tags ---
  const cleanResponse = (text) => {
    // Replaces everything between <thinking> and </thinking> with an empty string
    return text.replace(/<thinking>[\s\S]*?<\/thinking>/g, '').trim();
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    const endpoint = isLoginView ? '/token' : '/register';
    const payload = isLoginView 
      ? new URLSearchParams({ username: email, password: password }) 
      : JSON.stringify({ email, password });

    const headers = isLoginView 
      ? { 'Content-Type': 'application/x-www-form-urlencoded' }
      : { 'Content-Type': 'application/json' };

    try {
      const res = await fetch(`${API_URL}${endpoint}`, { method: 'POST', headers, body: payload });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Auth failed');

      if (isLoginView) {
        setToken(data.access_token);
        localStorage.setItem('token', data.access_token);
      } else {
        alert("Registered! Please log in.");
        setIsLoginView(true);
      }
    } catch (err) {
      alert(err.message);
    }
  };

  const logout = () => {
    setToken(null);
    localStorage.removeItem('token');
    setMessages([]);
    setSessionId(null);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          ...(sessionId && { 'X-Session-ID': sessionId })
        },
        body: JSON.stringify({ query: userMsg.content, history: messages })
      });

      if (res.status === 401) {
        logout();
        return;
      }

      const newSessionId = res.headers.get('X-Session-ID');
      if (newSessionId) setSessionId(newSessionId);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botMsg = { role: 'bot', content: '' };
      
      setMessages(prev => [...prev, botMsg]); 

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        botMsg.content += chunk;
        
        setMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1] = { ...botMsg };
          return newMsgs;
        });
      }

    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'bot', content: "Error connecting to server." }]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) {
    return (
      <div className="auth-container">
        <div className="glass-card auth-box">
          <h2>{isLoginView ? "Welcome Back" : "Create Account"}</h2>
          <form onSubmit={handleAuth}>
            <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} required />
            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
            <button type="submit">{isLoginView ? "Login" : "Register"}</button>
          </form>
          <p onClick={() => setIsLoginView(!isLoginView)}>
            {isLoginView ? "Need an account? Register" : "Have an account? Login"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <header className="glass-header">
        <h1>🇻🇳 VietBot AI</h1>
        <button onClick={logout} className="logout-btn">Logout</button>
      </header>

      <div className="messages-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role} fade-in`}>
            <div className="bubble">
              {/* Use ReactMarkdown to render bold, lists, etc. */}
              <ReactMarkdown>
                {msg.role === 'bot' ? cleanResponse(msg.content) : msg.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        {isLoading && <div className="message bot"><div className="bubble loading">typing...</div></div>}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={sendMessage} className="input-area glass-input">
        <input 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          placeholder="Ask about travel in Vietnam..." 
        />
        <button type="submit" disabled={isLoading}>
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>
        </button>
      </form>
    </div>
  );
}

export default App;