import { useState, useEffect, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useUser } from '@/lib/auth';
import { useWebSocket } from '@/lib/websocket';
import { Loader2, Users, MusicIcon, RefreshCw, Share2 } from 'lucide-react';
import { GENRE_OPTIONS } from '@/lib/constants';

/**
 * CollabLab Component
 * 
 * A real-time collaboration environment for music creation
 * Inspired by the Suno.com interface but with enhanced collaboration features
 */
export function CollabLab() {
  const { user } = useUser();
  const { toast } = useToast();
  
  // State
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [activeTab, setActiveTab] = useState('lyrics');
  const [isLoading, setIsLoading] = useState(true);
  const [view, setView] = useState('browse'); // 'browse', 'create', 'session'
  const [lyrics, setLyrics] = useState('');
  const [genre, setGenre] = useState('');
  const [styleOptions, setStyleOptions] = useState([]);
  const [selectedStyles, setSelectedStyles] = useState([]);
  const [participants, setParticipants] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  
  // WebSocket connection
  const {
    connect,
    disconnect,
    sendMessage,
    lastMessage,
    readyState
  } = useWebSocket();
  
  // Refs
  const chatEndRef = useRef(null);
  
  // Connect to websocket when session is active
  useEffect(() => {
    if (activeSession && user) {
      const wsUrl = `ws://${window.location.host}/api/v1/ws/collab/${activeSession.id}?user_id=${user.id}`;
      connect(wsUrl);
      
      return () => {
        disconnect();
      };
    }
  }, [activeSession, user]);
  
  // Handle incoming websocket messages
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      
      switch (data.type) {
        case 'session_state':
          // Update session state from server
          setLyrics(data.content.lyrics);
          setGenre(data.content.genre);
          setSelectedStyles(data.content.style_tags);
          setParticipants(data.content.participants);
          break;
          
        case 'lyrics_update':
          setLyrics(data.content);
          break;
          
        case 'genre_update':
          setGenre(data.content);
          break;
          
        case 'style_update':
          setSelectedStyles(data.content);
          break;
          
        case 'chat':
          setChatMessages(prev => [...prev, data]);
          // Scroll to bottom of chat
          chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
          break;
          
        case 'user_joined':
        case 'user_left':
          // Add system message to chat
          setChatMessages(prev => [...prev, data]);
          // Update participants list
          fetchSessionDetails(activeSession.id);
          break;
          
        case 'generation_started':
          setIsGenerating(true);
          toast({
            title: 'Generation Started',
            description: 'Your music is being created...',
          });
          // Add system message to chat
          setChatMessages(prev => [...prev, {
            type: 'chat',
            content: `Music generation started by ${data.sender_name || data.sender_id}`,
            sender_id: 'system',
            timestamp: new Date().toISOString(),
          }]);
          break;
          
        case 'generation_progress':
          // Update progress info in chat
          setChatMessages(prev => [...prev, {
            type: 'chat',
            content: `Generation progress: ${data.content.progress}%`,
            sender_id: 'system',
            timestamp: new Date().toISOString(),
          }]);
          break;
          
        case 'generation_complete':
          setIsGenerating(false);
          // Update active session with new output URL
          if (data.content && data.content.output_url) {
            setActiveSession(prev => ({
              ...prev,
              output_url: data.content.output_url
            }));
            
            // Automatically switch to the output tab
            setActiveTab('output');
          }
          
          toast({
            title: 'Music Generated!',
            description: 'Your collaborative creation is ready to play',
          });
          
          // Add system message to chat
          setChatMessages(prev => [...prev, {
            type: 'chat',
            content: 'Music generation complete! Check the Output tab to listen.',
            sender_id: 'system',
            timestamp: new Date().toISOString(),
          }]);
          break;
          
        case 'generation_error':
          setIsGenerating(false);
          toast({
            title: 'Generation Error',
            description: data.content || 'An error occurred during music generation',
            variant: 'destructive'
          });
          
          // Add error message to chat
          setChatMessages(prev => [...prev, {
            type: 'chat',
            content: `Error: ${data.content || 'Failed to generate music'}`,
            sender_id: 'system',
            timestamp: new Date().toISOString(),
          }]);
          break;
      }
    }
  }, [lastMessage]);
  
  // Load sessions on component mount
  useEffect(() => {
    fetchSessions();
    
    // Fetch style options
    fetchStyleOptions();
  }, []);
  
  // Auto-scroll chat to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);
  
  // Fetch available sessions
  const fetchSessions = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/collab-lab/sessions');
      const data = await response.json();
      setSessions(data);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load collaboration sessions',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch detailed session info
  const fetchSessionDetails = async (sessionId) => {
    try {
      const response = await fetch(`/api/v1/collab-lab/sessions/${sessionId}`);
      const data = await response.json();
      setActiveSession(data);
      setParticipants(data.participants);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load session details',
        variant: 'destructive',
      });
    }
  };
  
  // Fetch available style options
  const fetchStyleOptions = async () => {
    try {
      // In a real implementation, this would fetch from an API
      // For now, using mock data
      setStyleOptions([
        { id: 'electronic', name: 'Electronic' },
        { id: 'acoustic', name: 'Acoustic' },
        { id: 'orchestral', name: 'Orchestral' },
        { id: 'cinematic', name: 'Cinematic' },
        { id: 'lofi', name: 'Lo-Fi' },
        { id: 'ambient', name: 'Ambient' },
        { id: 'upbeat', name: 'Upbeat' },
        { id: 'melancholic', name: 'Melancholic' },
      ]);
    } catch (error) {
      console.error('Failed to load style options', error);
    }
  };
  
  // Create a new session
  const createSession = async () => {
    if (!newSessionName.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter a session name',
        variant: 'destructive',
      });
      return;
    }
    
    setIsCreatingSession(true);
    
    try {
      const response = await fetch('/api/v1/collab-lab/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newSessionName,
          creator_id: user.id,
          is_public: true,
        }),
      });
      
      const data = await response.json();
      
      // Navigate to new session
      setActiveSession(data);
      setView('session');
      
      toast({
        title: 'Success',
        description: 'Collaboration session created!',
      });
      
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to create session',
        variant: 'destructive',
      });
    } finally {
      setIsCreatingSession(false);
    }
  };
  
  // Join an existing session
  const joinSession = (session) => {
    setActiveSession(session);
    setView('session');
  };
  
  // Update lyrics and broadcast change
  const updateLyrics = (newLyrics) => {
    setLyrics(newLyrics);
    
    // Broadcast change to all participants
    if (readyState === 1) { // WebSocket.OPEN
      sendMessage(JSON.stringify({
        type: 'lyrics_update',
        content: newLyrics,
        sender_id: user.id,
      }));
    }
  };
  
  // Update genre and broadcast change
  const updateGenre = (newGenre) => {
    setGenre(newGenre);
    
    // Broadcast change to all participants
    if (readyState === 1) { // WebSocket.OPEN
      sendMessage(JSON.stringify({
        type: 'genre_update',
        content: newGenre,
        sender_id: user.id,
      }));
    }
  };
  
  // Toggle style selection and broadcast change
  const toggleStyle = (styleId) => {
    let newStyles;
    
    if (selectedStyles.includes(styleId)) {
      newStyles = selectedStyles.filter(id => id !== styleId);
    } else {
      newStyles = [...selectedStyles, styleId];
    }
    
    setSelectedStyles(newStyles);
    
    // Broadcast change to all participants
    if (readyState === 1) { // WebSocket.OPEN
      sendMessage(JSON.stringify({
        type: 'style_update',
        content: newStyles,
        sender_id: user.id,
      }));
    }
  };
  
  // Send chat message
  const sendChatMessage = () => {
    if (!chatInput.trim()) return;
    
    const message = {
      type: 'chat',
      content: chatInput,
      sender_id: user.id,
      sender_name: user.name,
      timestamp: new Date().toISOString(),
    };
    
    // Add to local state
    setChatMessages(prev => [...prev, message]);
    
    // Send to all participants
    if (readyState === 1) { // WebSocket.OPEN
      sendMessage(JSON.stringify(message));
    }
    
    // Clear input
    setChatInput('');
  };
  
  // Start music generation
  const generateMusic = () => {
    if (!lyrics || !genre) {
      toast({
        title: 'Error',
        description: 'Please add lyrics and select a genre',
        variant: 'destructive',
      });
      return;
    }
    
    // Send generation request
    if (readyState === 1) { // WebSocket.OPEN
      sendMessage(JSON.stringify({
        type: 'generate',
        content: {
          lyrics,
          genre,
          style_tags: selectedStyles,
        },
        sender_id: user.id,
      }));
    }
    
    toast({
      title: 'Generating Music',
      description: 'Your collaborative track will be ready soon!',
    });
  };
  
  // Share session link
  const shareSession = () => {
    const sessionUrl = `${window.location.origin}/collab-lab/${activeSession.id}`;
    
    navigator.clipboard.writeText(sessionUrl)
      .then(() => {
        toast({
          title: 'Link Copied!',
          description: 'Share this link with friends to collaborate',
        });
      })
      .catch(() => {
        toast({
          title: 'Failed to copy',
          description: sessionUrl,
          variant: 'destructive',
        });
      });
  };
  
  // Render browse sessions view
  const renderBrowseView = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Collaboration Sessions</h2>
        <Button onClick={() => setView('create')}>Create New Session</Button>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : sessions.length === 0 ? (
        <div className="text-center py-12 border rounded-lg bg-muted/50">
          <p className="text-lg mb-4">No active sessions found</p>
          <Button onClick={() => setView('create')}>Create Your First Session</Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sessions.map(session => (
            <Card key={session.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <span>{session.name}</span>
                  <Badge variant="outline" className="ml-2">
                    <Users className="h-3 w-3 mr-1" />
                    {session.participants.length}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Created by {session.creator_id === user?.id ? 'you' : 'another user'}
                </p>
                {session.genre && (
                  <Badge className="mr-2 mb-2">{session.genre}</Badge>
                )}
                {session.style_tags.slice(0, 3).map(tag => (
                  <Badge key={tag} variant="outline" className="mr-2 mb-2">{tag}</Badge>
                ))}
              </CardContent>
              <CardFooter>
                <Button 
                  onClick={() => joinSession(session)}
                  className="w-full"
                >
                  Join Session
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
      
      <div className="flex justify-center mt-4">
        <Button 
          variant="outline" 
          onClick={fetchSessions}
          className="flex items-center"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh List
        </Button>
      </div>
    </div>
  );
  
  // Render create session view
  const renderCreateView = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Create Collaboration Session</h2>
        <Button variant="outline" onClick={() => setView('browse')}>Back to Sessions</Button>
      </div>
      
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="session-name" className="block text-sm font-medium mb-2">
                Session Name
              </label>
              <Input
                id="session-name"
                placeholder="My Awesome Track"
                value={newSessionName}
                onChange={(e) => setNewSessionName(e.target.value)}
              />
            </div>
            
            <div className="flex justify-end">
              <Button
                onClick={createSession}
                disabled={isCreatingSession || !newSessionName.trim()}
              >
                {isCreatingSession && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Create Session
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
  
  // Render active session view
  const renderSessionView = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center">
          <h2 className="text-2xl font-bold">{activeSession?.name}</h2>
          <Badge variant="outline" className="ml-3">
            <Users className="h-3 w-3 mr-1" />
            {participants.length} collaborator{participants.length !== 1 ? 's' : ''}
          </Badge>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={shareSession}
          >
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setView('browse')}
          >
            Exit Session
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main workspace - 2/3 width on desktop */}
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="lyrics" value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="w-full">
              <TabsTrigger value="lyrics" className="flex-1">Lyrics</TabsTrigger>
              <TabsTrigger value="settings" className="flex-1">Style & Settings</TabsTrigger>
              <TabsTrigger value="output" className="flex-1">Output</TabsTrigger>
            </TabsList>
            
            <TabsContent value="lyrics" className="pt-4">
              <Card>
                <CardContent className="pt-6">
                  <Textarea
                    placeholder="Write your lyrics here..."
                    className="min-h-[300px] font-mono"
                    value={lyrics}
                    onChange={(e) => updateLyrics(e.target.value)}
                  />
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="settings" className="pt-4">
              <Card>
                <CardContent className="pt-6 space-y-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Genre
                    </label>
                    <Select value={genre} onValueChange={updateGenre}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a genre" />
                      </SelectTrigger>
                      <SelectContent>
                        {GENRE_OPTIONS.map(genreOption => (
                          <SelectItem key={genreOption.id} value={genreOption.id}>
                            {genreOption.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Style Tags (select multiple)
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {styleOptions.map(style => (
                        <Badge
                          key={style.id}
                          variant={selectedStyles.includes(style.id) ? "default" : "outline"}
                          className="cursor-pointer"
                          onClick={() => toggleStyle(style.id)}
                        >
                          {style.name}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="output" className="pt-4">
              <Card>
                <CardContent className="pt-6">
                  {isGenerating ? (
                    <div className="flex flex-col items-center justify-center py-12">
                      <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                      <p>Generating your collaborative masterpiece...</p>
                    </div>
                  ) : activeSession?.output_url ? (
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-2">
                          Preview
                        </label>
                        <audio 
                          controls 
                          className="w-full" 
                          src={activeSession.output_url}
                        />
                      </div>
                      <div className="flex justify-end">
                        <Button 
                          variant="outline"
                          onClick={() => window.open(activeSession.output_url, '_blank')}
                        >
                          Download
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12">
                      <MusicIcon className="h-16 w-16 text-muted-foreground mb-4" />
                      <p className="mb-6">No output generated yet</p>
                      <Button 
                        onClick={generateMusic}
                        disabled={!lyrics || !genre}
                      >
                        Generate Music
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Chat sidebar - 1/3 width on desktop */}
        <div>
          <Card className="h-[600px] flex flex-col">
            <CardHeader>
              <CardTitle>Chat</CardTitle>
            </CardHeader>
            <div className="flex-1 overflow-y-auto px-4">
              {chatMessages.length === 0 ? (
                <div className="h-full flex items-center justify-center text-center text-muted-foreground">
                  <p>No messages yet.<br />Start the conversation!</p>
                </div>
              ) : (
                <div className="space-y-4 py-2">
                  {chatMessages.map((msg, index) => (
                    <div 
                      key={index} 
                      className={`flex ${msg.sender_id === user?.id ? 'justify-end' : 'justify-start'}`}
                    >
                      <div 
                        className={`max-w-[80%] rounded-lg px-3 py-2 ${
                          msg.sender_id === 'system' 
                            ? 'bg-muted text-center w-full' 
                            : msg.sender_id === user?.id 
                              ? 'bg-primary text-primary-foreground' 
                              : 'bg-secondary'
                        }`}
                      >
                        {msg.sender_id !== 'system' && msg.sender_id !== user?.id && (
                          <p className="text-xs font-medium mb-1">{msg.sender_name || msg.sender_id}</p>
                        )}
                        <p>{msg.content}</p>
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>
            <div className="p-4 border-t">
              <div className="flex gap-2">
                <Input
                  placeholder="Type a message..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                />
                <Button onClick={sendChatMessage}>Send</Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
  
  // Render appropriate view based on state
  const renderContent = () => {
    switch (view) {
      case 'browse':
        return renderBrowseView();
      case 'create':
        return renderCreateView();
      case 'session':
        return renderSessionView();
      default:
        return renderBrowseView();
    }
  };
  
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Beat Addicts Collab Lab</h1>
      {renderContent()}
    </div>
  );
}

export default CollabLab;
