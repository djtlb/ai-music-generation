import { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useMusicPipeline, StyleConfig } from '@/hooks/useMusicPipeline';
import { Music, Play, Square, Download, Settings, Wand2, CheckCircle2, AlertCircle, Clock } from 'lucide-react';

export function IntegratedCompositionWorkflow() {
  const {
    pipelineState,
    currentProjectData,
    createProject,
    loadProject,
    runFullPipeline,
    projects,
    setProcessing,
    clearError,
  } = useMusicPipeline();

  const [newProjectForm, setNewProjectForm] = useState({
    name: '',
    genre: 'pop',
    subgenre: '',
    energy: 0.7,
    mood: 'uplifting',
    tempo: 120,
    key: 'C',
    timeSignature: '4/4',
    lyricsTheme: '',
  });

  const [isCreatingProject, setIsCreatingProject] = useState(false);

  const handleCreateProject = useCallback(async () => {
    if (!newProjectForm.name.trim()) return;

    setIsCreatingProject(true);
    
    try {
      const styleConfig: StyleConfig = {
        genre: newProjectForm.genre,
        subgenre: newProjectForm.subgenre || undefined,
        energy: newProjectForm.energy,
        mood: newProjectForm.mood,
        tempo: newProjectForm.tempo,
        key: newProjectForm.key,
        timeSignature: newProjectForm.timeSignature,
      };

      await runFullPipeline(newProjectForm.name, styleConfig, {
        lyricsTheme: newProjectForm.lyricsTheme || undefined,
        autoAdvance: true,
      });

      // Reset form
      setNewProjectForm({
        name: '',
        genre: 'pop',
        subgenre: '',
        energy: 0.7,
        mood: 'uplifting',
        tempo: 120,
        key: 'C',
        timeSignature: '4/4',
        lyricsTheme: '',
      });
    } catch (error) {
      console.error('Failed to create project:', error);
    } finally {
      setIsCreatingProject(false);
    }
  }, [newProjectForm, runFullPipeline]);

  const handleLoadProject = useCallback((projectId: string) => {
    loadProject(projectId);
  }, [loadProject]);

  const getStageStatus = (stage: string) => {
    const projectData = currentProjectData;
    if (!projectData) return 'not-started';
    
    const hasError = stage in pipelineState.errors;
    if (hasError) return 'error';
    
    const isProcessing = pipelineState.isProcessing[stage as keyof typeof pipelineState.isProcessing];
    if (isProcessing) return 'processing';
    
    switch (stage) {
      case 'lyrics':
        return projectData.lyrics ? 'completed' : 'not-started';
      case 'arrangement':
        return projectData.arrangement ? 'completed' : 'not-started';
      case 'composition':
        return projectData.composition ? 'completed' : 'not-started';
      case 'soundDesign':
        return projectData.soundDesign ? 'completed' : 'not-started';
      case 'mixMaster':
        return projectData.mixMaster ? 'completed' : 'not-started';
      default:
        return 'not-started';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <div className="h-4 w-4 rounded-full border-2 border-gray-300" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'processing':
        return 'bg-blue-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };

  const calculateProgress = () => {
    if (!currentProjectData) return 0;
    
    let completedStages = 0;
    const totalStages = 5;
    
    if (currentProjectData.lyrics) completedStages++;
    if (currentProjectData.arrangement) completedStages++;
    if (currentProjectData.composition) completedStages++;
    if (currentProjectData.soundDesign) completedStages++;
    if (currentProjectData.mixMaster) completedStages++;
    
    return (completedStages / totalStages) * 100;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Integrated Composition Workflow</h2>
          <p className="text-muted-foreground">
            Create complete songs from concept to master with AI assistance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Music className="h-8 w-8 text-primary" />
          <Wand2 className="h-6 w-6 text-blue-500" />
        </div>
      </div>

      <Tabs defaultValue="workspace" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="workspace">Workspace</TabsTrigger>
          <TabsTrigger value="new-project">New Project</TabsTrigger>
          <TabsTrigger value="projects">Projects</TabsTrigger>
        </TabsList>

        {/* Workspace Tab */}
        <TabsContent value="workspace" className="space-y-6">
          {currentProjectData ? (
            <>
              {/* Current Project Header */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {currentProjectData.project.name}
                        <Badge variant="outline">
                          {currentProjectData.project.styleConfig.genre}
                        </Badge>
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        {currentProjectData.project.styleConfig.key} • {currentProjectData.project.styleConfig.tempo} BPM • {currentProjectData.project.styleConfig.timeSignature}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">{Math.round(calculateProgress())}%</div>
                      <div className="text-sm text-muted-foreground">Complete</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Progress value={calculateProgress()} className="w-full" />
                </CardContent>
              </Card>

              {/* Pipeline Status */}
              <Card>
                <CardHeader>
                  <CardTitle>Pipeline Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    {[
                      { key: 'lyrics', label: 'Lyrics', data: currentProjectData.lyrics },
                      { key: 'arrangement', label: 'Arrangement', data: currentProjectData.arrangement },
                      { key: 'composition', label: 'Composition', data: currentProjectData.composition },
                      { key: 'soundDesign', label: 'Sound Design', data: currentProjectData.soundDesign },
                      { key: 'mixMaster', label: 'Mix & Master', data: currentProjectData.mixMaster },
                    ].map((stage, index) => {
                      const status = getStageStatus(stage.key);
                      const isActive = index === 0 || getStageStatus([
                        'lyrics', 'arrangement', 'composition', 'soundDesign', 'mixMaster'
                      ][index - 1]) === 'completed';
                      
                      return (
                        <div key={stage.key} className="relative">
                          <Card className={`transition-all ${isActive ? 'ring-2 ring-primary' : 'opacity-60'}`}>
                            <CardContent className="pt-6">
                              <div className="flex flex-col items-center text-center space-y-2">
                                {getStatusIcon(status)}
                                <div className="font-medium text-sm">{stage.label}</div>
                                <Badge variant={status === 'completed' ? 'default' : 'secondary'} className="text-xs">
                                  {status === 'completed' ? 'Done' : 
                                   status === 'processing' ? 'Processing...' :
                                   status === 'error' ? 'Error' : 'Waiting'}
                                </Badge>
                                {stage.data && (
                                  <div className="text-xs text-muted-foreground">
                                    ID: {stage.data.id.split('-')[1]}
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                          
                          {/* Connection line */}
                          {index < 4 && (
                            <div className={`hidden md:block absolute top-1/2 -right-2 w-4 h-0.5 ${getStatusColor(status)}`} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Stage Details */}
              {currentProjectData.arrangement && (
                <Card>
                  <CardHeader>
                    <CardTitle>Current Arrangement</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <Label className="text-sm font-medium">Genre</Label>
                        <p className="text-sm">{currentProjectData.arrangement.genre}</p>
                      </div>
                      <div>
                        <Label className="text-sm font-medium">Tempo</Label>
                        <p className="text-sm">{currentProjectData.arrangement.bpm} BPM</p>
                      </div>
                      <div>
                        <Label className="text-sm font-medium">Key</Label>
                        <p className="text-sm">{currentProjectData.arrangement.key}</p>
                      </div>
                      <div>
                        <Label className="text-sm font-medium">Duration</Label>
                        <p className="text-sm">{currentProjectData.arrangement.totalBars} bars</p>
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <Label className="text-sm font-medium mb-2 block">Song Structure</Label>
                      <div className="flex flex-wrap gap-2">
                        {currentProjectData.arrangement.structure.map((section, idx) => (
                          <Badge key={idx} variant="outline">
                            {section.section} ({section.startBar}-{section.endBar})
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Composition Details */}
              {currentProjectData.composition && (
                <Card>
                  <CardHeader>
                    <CardTitle>Current Composition</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div>
                          <Label className="text-sm font-medium">Tracks</Label>
                          <p className="text-sm">{currentProjectData.composition.tracks.length}</p>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Key</Label>
                          <p className="text-sm">{currentProjectData.composition.key}</p>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Tempo</Label>
                          <p className="text-sm">{currentProjectData.composition.tempo} BPM</p>
                        </div>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium mb-2 block">Instruments</Label>
                        <div className="flex flex-wrap gap-2">
                          {currentProjectData.composition.tracks.map((track, idx) => (
                            <Badge key={idx} variant="secondary">
                              {track.instrument}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium mb-2 block">Chord Progression</Label>
                        <div className="flex flex-wrap gap-2">
                          {currentProjectData.composition.chordProgression.map((chord, idx) => (
                            <Badge key={idx} variant="outline">
                              {chord}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Control Panel */}
              <Card>
                <CardHeader>
                  <CardTitle>Controls</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    <Button size="sm" disabled>
                      <Play className="w-4 h-4 mr-2" />
                      Preview
                    </Button>
                    <Button size="sm" variant="outline" disabled>
                      <Download className="w-4 h-4 mr-2" />
                      Export MIDI
                    </Button>
                    <Button size="sm" variant="outline" disabled>
                      <Download className="w-4 h-4 mr-2" />
                      Export Audio
                    </Button>
                    <Button size="sm" variant="outline" disabled>
                      <Settings className="w-4 h-4 mr-2" />
                      Advanced Settings
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center py-12">
                  <Music className="mx-auto h-12 w-12 text-muted-foreground" />
                  <h3 className="mt-4 text-lg font-semibold">No Project Selected</h3>
                  <p className="text-muted-foreground">
                    Create a new project or load an existing one to get started
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* New Project Tab */}
        <TabsContent value="new-project" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create New Project</CardTitle>
              <p className="text-muted-foreground">
                Set up your project parameters and let AI generate a complete song
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="project-name">Project Name</Label>
                  <Input
                    id="project-name"
                    placeholder="My New Song"
                    value={newProjectForm.name}
                    onChange={(e) => setNewProjectForm(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lyrics-theme">Lyrics Theme (Optional)</Label>
                  <Input
                    id="lyrics-theme"
                    placeholder="Love, Adventure, Freedom..."
                    value={newProjectForm.lyricsTheme}
                    onChange={(e) => setNewProjectForm(prev => ({ ...prev, lyricsTheme: e.target.value }))}
                  />
                </div>
              </div>

              {/* Style Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="genre">Genre</Label>
                  <Select
                    value={newProjectForm.genre}
                    onValueChange={(value) => setNewProjectForm(prev => ({ ...prev, genre: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pop">Pop</SelectItem>
                      <SelectItem value="rock">Rock</SelectItem>
                      <SelectItem value="jazz">Jazz</SelectItem>
                      <SelectItem value="electronic">Electronic</SelectItem>
                      <SelectItem value="folk">Folk</SelectItem>
                      <SelectItem value="classical">Classical</SelectItem>
                      <SelectItem value="hip-hop">Hip-Hop</SelectItem>
                      <SelectItem value="country">Country</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mood">Mood</Label>
                  <Select
                    value={newProjectForm.mood}
                    onValueChange={(value) => setNewProjectForm(prev => ({ ...prev, mood: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="uplifting">Uplifting</SelectItem>
                      <SelectItem value="melancholic">Melancholic</SelectItem>
                      <SelectItem value="energetic">Energetic</SelectItem>
                      <SelectItem value="calm">Calm</SelectItem>
                      <SelectItem value="mysterious">Mysterious</SelectItem>
                      <SelectItem value="romantic">Romantic</SelectItem>
                      <SelectItem value="aggressive">Aggressive</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="energy-slider">Energy Level: {Math.round(newProjectForm.energy * 100)}%</Label>
                  <input
                    id="energy-slider"
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={newProjectForm.energy}
                    onChange={(e) => setNewProjectForm(prev => ({ ...prev, energy: parseFloat(e.target.value) }))}
                    className="w-full"
                    title={`Energy Level: ${Math.round(newProjectForm.energy * 100)}%`}
                  />
                </div>
              </div>

              {/* Musical Parameters */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="tempo">Tempo (BPM)</Label>
                  <Input
                    id="tempo"
                    type="number"
                    min="60"
                    max="200"
                    value={newProjectForm.tempo}
                    onChange={(e) => setNewProjectForm(prev => ({ ...prev, tempo: parseInt(e.target.value) || 120 }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="key">Key</Label>
                  <Select
                    value={newProjectForm.key}
                    onValueChange={(value) => setNewProjectForm(prev => ({ ...prev, key: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="C">C</SelectItem>
                      <SelectItem value="C#">C#</SelectItem>
                      <SelectItem value="D">D</SelectItem>
                      <SelectItem value="D#">D#</SelectItem>
                      <SelectItem value="E">E</SelectItem>
                      <SelectItem value="F">F</SelectItem>
                      <SelectItem value="F#">F#</SelectItem>
                      <SelectItem value="G">G</SelectItem>
                      <SelectItem value="G#">G#</SelectItem>
                      <SelectItem value="A">A</SelectItem>
                      <SelectItem value="A#">A#</SelectItem>
                      <SelectItem value="B">B</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="time-signature">Time Signature</Label>
                  <Select
                    value={newProjectForm.timeSignature}
                    onValueChange={(value) => setNewProjectForm(prev => ({ ...prev, timeSignature: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="4/4">4/4</SelectItem>
                      <SelectItem value="3/4">3/4</SelectItem>
                      <SelectItem value="6/8">6/8</SelectItem>
                      <SelectItem value="2/4">2/4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button 
                onClick={handleCreateProject} 
                disabled={!newProjectForm.name.trim() || isCreatingProject}
                className="w-full"
              >
                {isCreatingProject ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Creating Project...
                  </>
                ) : (
                  <>
                    <Wand2 className="w-4 h-4 mr-2" />
                    Create Complete Song
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Projects Tab */}
        <TabsContent value="projects" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Your Projects</CardTitle>
              <p className="text-muted-foreground">
                Load and manage your existing projects
              </p>
            </CardHeader>
            <CardContent>
              {projects.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {projects.map((project) => (
                    <Card key={project?.id} className="cursor-pointer hover:shadow-md transition-shadow">
                      <CardContent className="pt-6">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <h3 className="font-semibold">{project?.name}</h3>
                            {currentProjectData?.project.id === project?.id && (
                              <Badge variant="default">Active</Badge>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {project?.styleConfig.genre} • {project?.styleConfig.tempo} BPM
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {project?.styleConfig.key} • {project?.styleConfig.mood}
                          </div>
                          <Button 
                            size="sm" 
                            className="w-full"
                            onClick={() => handleLoadProject(project?.id || '')}
                            disabled={currentProjectData?.project.id === project?.id}
                          >
                            {currentProjectData?.project.id === project?.id ? 'Current Project' : 'Load Project'}
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Music className="mx-auto h-12 w-12 text-muted-foreground" />
                  <h3 className="mt-4 text-lg font-semibold">No Projects Yet</h3>
                  <p className="text-muted-foreground">
                    Create your first project to get started
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
