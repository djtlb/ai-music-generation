import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * WebSocket hook for real-time communication
 * 
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Lifecycle management (connect, disconnect, send)
 * - Status tracking (connecting, open, closing, closed)
 */
export const useWebSocket = () => {
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(-1); // -1 = not initialized
  
  // Keep socket reference in a ref to avoid re-initializing
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000; // ms
  
  // Connect to WebSocket
  const connect = useCallback((url) => {
    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Clean up existing socket
    if (socketRef.current) {
      socketRef.current.close();
    }
    
    // Create new WebSocket
    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;
      
      // Update ready state
      setReadyState(socket.readyState);
      
      // Set up event handlers
      socket.addEventListener('open', () => {
        setReadyState(WebSocket.OPEN);
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
        console.log('WebSocket connection established');
      });
      
      socket.addEventListener('message', (event) => {
        setLastMessage(event);
      });
      
      socket.addEventListener('close', (event) => {
        setReadyState(WebSocket.CLOSED);
        console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
        
        // Attempt to reconnect if the close wasn't intentional (code 1000)
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(1.5, reconnectAttemptsRef.current),
            30000 // Max delay of 30 seconds
          );
          
          console.log(`Attempting to reconnect in ${delay}ms...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current += 1;
            connect(url);
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.error('Max reconnect attempts reached');
        }
      });
      
      socket.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
      });
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, []);
  
  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    // Clear any reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Close socket if it exists
    if (socketRef.current) {
      socketRef.current.close(1000, 'Disconnected by user');
      socketRef.current = null;
    }
    
    setReadyState(WebSocket.CLOSED);
  }, []);
  
  // Send message through WebSocket
  const sendMessage = useCallback((message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(message);
      return true;
    }
    return false;
  }, []);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (socketRef.current) {
        socketRef.current.close(1000, 'Component unmounted');
      }
    };
  }, []);
  
  return {
    connect,
    disconnect,
    sendMessage,
    lastMessage,
    readyState,
    READY_STATE: {
      CONNECTING: WebSocket.CONNECTING,
      OPEN: WebSocket.OPEN,
      CLOSING: WebSocket.CLOSING,
      CLOSED: WebSocket.CLOSED
    }
  };
};

export default useWebSocket;
