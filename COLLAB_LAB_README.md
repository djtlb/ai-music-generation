# Beat Addicts - Real-time Music Collaboration Platform

## Collab Lab Feature

The Beat Addicts Collab Lab enables real-time music collaboration, similar to Suno.com's interface but with enhanced collaborative capabilities. Multiple users can work together on music projects simultaneously, each contributing to different aspects of the music creation process.

### Key Features

#### 1. Real-time Collaboration
- Multiple users can join a collaboration session
- Changes to lyrics, genre, and style settings are instantly synced to all participants
- Built-in chat system for communication during collaboration
- Session persistence for returning to work-in-progress projects

#### 2. Creative Workflow
- Collaborative lyric writing with real-time updates
- Genre and style selection with instant feedback
- AI-powered music generation based on collaborative inputs
- Shared listening experience for generated music

#### 3. User Experience
- Clean, modern interface inspired by Suno.com
- Intuitive session management (create, join, leave)
- Clear indication of who's currently in the session
- Real-time notifications for user actions

## Technical Implementation

### Backend

The backend is implemented using FastAPI with WebSockets for real-time communication:

- **WebSocket Protocol**: Enables bidirectional, real-time communication
- **Session Management**: Tracks active collaboration sessions and participants
- **Message Handling**: Routes different message types (lyrics updates, genre changes, etc.)
- **State Synchronization**: Ensures all clients have the same view of the project

### Frontend

The frontend is built with React and TailwindCSS:

- **React Components**: Modular components for different parts of the collaboration UI
- **WebSocket Hook**: Custom hook for managing WebSocket connections
- **Real-time Updates**: State management that responds to incoming messages
- **Responsive Design**: Works across desktop and mobile devices

## Getting Started

### Running the Backend

```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```

### Running the Frontend

```bash
npm install
npm run dev
```

### Usage

1. Open the application in your browser
2. Create a new collaboration session or join an existing one
3. Start collaborating on lyrics, genre, and style
4. Generate music together
5. Download the final product

## Future Enhancements

- Audio waveform visualization during playback
- Multiple audio tracks support
- Stem-level mixing controls
- Version history with branching
- Expanded genre and style options
- Direct integration with social platforms
- Mobile app for collaboration on the go

---

© 2025 Beat Addicts - AI Music Generation Platform
