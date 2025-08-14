# AI Music Composer - Product Requirements Document

An AI-powered music composition assistant that helps users create original musical ideas through intelligent prompts, lyric generation, and compositional guidance.

**Experience Qualities**: 
1. **Creative** - Inspires musical exploration and artistic expression through intelligent suggestions
2. **Intuitive** - Makes music creation accessible to both beginners and experienced musicians
3. **Professional** - Provides serious compositional tools while maintaining ease of use

**Complexity Level**: Light Application (multiple features with basic state)
The app provides several interconnected features for music creation but maintains simplicity in user interaction and state management.

## Essential Features

### AI Lyric Generator
- **Functionality**: Generates original lyrics based on theme, mood, genre, and user prompts
- **Purpose**: Helps overcome writer's block and provides creative starting points for songwriting
- **Trigger**: User selects genre/mood and enters optional theme or keywords
- **Progression**: Select parameters → Enter theme → Generate lyrics → Edit/refine → Save/export
- **Success criteria**: Generates coherent, rhyming lyrics that match specified parameters

### Chord Progression Builder
- **Functionality**: Suggests chord progressions based on key, genre, and mood with audio preview
- **Purpose**: Provides harmonic foundation for songs and teaches music theory concepts
- **Trigger**: User selects key signature, genre, and desired complexity level
- **Progression**: Choose parameters → Generate progression → Preview audio → Modify chords → Save progression
- **Success criteria**: Creates musically sound progressions with working audio playback

### Song Structure Planner
- **Functionality**: Creates song arrangements with verse/chorus/bridge patterns and timing
- **Purpose**: Helps organize musical ideas into complete song structures
- **Trigger**: User selects song type and desired length/complexity
- **Progression**: Choose template → Customize sections → Set timing → Export structure plan
- **Success criteria**: Generates logical song structures with proper pacing and transitions

### Composition History
- **Functionality**: Saves and organizes all generated lyrics, progressions, and structures
- **Purpose**: Allows users to revisit, combine, and develop their musical ideas over time
- **Trigger**: Automatic saving when content is generated or manually saved
- **Progression**: Create content → Auto-save → Browse history → Load previous work → Continue editing
- **Success criteria**: Reliably persists all user-generated content with easy retrieval

## Edge Case Handling

- **Empty prompts**: Provide example prompts and default suggestions when user input is minimal
- **Generation failures**: Graceful fallback with alternative suggestions and retry options
- **Audio playback issues**: Clear error messages with browser compatibility guidance
- **Storage limits**: Automatic cleanup of oldest items when storage approaches capacity
- **Offline usage**: Core functionality works without internet, with clear indicators for AI features

## Design Direction

The design should feel professional yet inspiring - balancing the precision of a music studio with the creativity of an artist's workspace. A clean, modern interface that doesn't overwhelm but invites exploration and experimentation.

## Color Selection

Complementary (opposite colors) - Using deep purple and warm gold to create sophisticated contrast that evokes both the mystery of creativity and the warmth of musical expression.

- **Primary Color**: Deep Purple (oklch(0.35 0.15 270)) - Represents creativity, artistry, and the depth of musical expression
- **Secondary Colors**: Charcoal Gray (oklch(0.25 0.02 270)) for supporting elements and subtle backgrounds
- **Accent Color**: Warm Gold (oklch(0.75 0.12 60)) - Attention-grabbing highlight for CTAs and important elements like play buttons
- **Foreground/Background Pairings**: 
  - Background (White oklch(0.98 0 0)): Dark Gray text (oklch(0.2 0.02 270)) - Ratio 12.1:1 ✓
  - Card (Light Gray oklch(0.96 0.01 270)): Dark Gray text (oklch(0.2 0.02 270)) - Ratio 11.3:1 ✓
  - Primary (Deep Purple oklch(0.35 0.15 270)): White text (oklch(0.98 0 0)) - Ratio 8.9:1 ✓
  - Accent (Warm Gold oklch(0.75 0.12 60)): Dark Gray text (oklch(0.2 0.02 270)) - Ratio 4.7:1 ✓

## Font Selection

Typography should convey both creativity and professionalism - modern sans-serif for clarity with distinctive character that reflects musical artistry.

- **Typographic Hierarchy**: 
  - H1 (App Title): Inter Bold/32px/tight letter spacing
  - H2 (Section Headers): Inter Semibold/24px/normal spacing  
  - H3 (Feature Labels): Inter Medium/18px/normal spacing
  - Body (Content): Inter Regular/16px/relaxed line height
  - Small (Labels): Inter Medium/14px/normal spacing

## Animations

Subtle and purposeful animations that enhance the creative flow without distraction - smooth transitions that feel like musical timing and rhythm.

- **Purposeful Meaning**: Animations should reflect musical concepts like rhythm, harmony, and flow - gentle easing that mirrors musical phrasing
- **Hierarchy of Movement**: Primary focus on content generation (lyric reveal, chord transitions), secondary on navigation and state changes

## Component Selection

- **Components**: 
  - Cards for each major feature section with subtle shadows
  - Tabs for switching between different creation modes
  - Select dropdowns for genre/key/mood selection with custom styling
  - Textarea for lyric display and editing with monospace font
  - Button variants for primary actions (generate), secondary (save), and play controls
  - Progress indicators for AI generation states
  - Accordion for organizing chord progressions and song structures

- **Customizations**: 
  - Audio waveform visualizer component for chord preview
  - Lyric formatter with verse/chorus highlighting
  - Chord diagram display component
  - Timeline component for song structure planning

- **States**: 
  - Generate buttons: Disabled during AI processing with loading spinner
  - Audio buttons: Clear play/pause states with visual feedback
  - Form inputs: Focus states with purple accent borders
  - Cards: Subtle hover effects with shadow increase

- **Icon Selection**: 
  - Music note icons for navigation
  - Play/pause for audio controls  
  - Wand/sparkles for AI generation
  - Save/bookmark for storing compositions
  - Clock for timing and tempo elements

- **Spacing**: 
  - Container padding: p-6 for main sections
  - Card spacing: gap-6 between major elements
  - Form spacing: gap-4 for related inputs
  - Button spacing: gap-2 for grouped actions

- **Mobile**: 
  - Responsive cards that stack vertically on mobile
  - Larger touch targets for audio controls
  - Collapsible sections for better mobile navigation
  - Bottom-sheet style modals for mobile interactions