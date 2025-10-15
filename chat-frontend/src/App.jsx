// src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown'; // <-- Import the markdown component
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! How can I help you plan your trip to Vietnam?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (!response.body) return;
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          const updatedLastMessage = { ...lastMessage, content: lastMessage.content + chunk };
          return [...prev.slice(0, -1), updatedLastMessage];
        });
      }
    } catch (error) {
      console.error('Failed to fetch:', error);
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        const updatedLastMessage = { ...lastMessage, content: 'Sorry, I ran into an error. Please try again.' };
        return [...prev.slice(0, -1), updatedLastMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>✈️ Vietnam AI Travel Assistant</h1>
      </div>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {/* --- THIS IS THE FIX --- */}
            <ReactMarkdown>{msg.content}</ReactMarkdown>
            {/* -------------------- */}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about travel in Vietnam..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default App;