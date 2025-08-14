import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { useKV } from '@github/spark/hooks';
import { toast } from "sonner";
import { Play, Download, Shuffle, Settings, Brain, Clock } from "@phosphor-icons/react";

interface Section {
  type: string;
  start_bar: number;
  length_bars: number;
}

interface ArrangementResult {
  style: string;
  tempo: number;
  duration_bars: number;
  sections: Section[];
  generation_params?: {
    temperature: number;
    top_k: number;
    top_p: number;
  };
}

export function ArrangementTransformerDemo() {
  // Generation parameters
  const [style, setStyle] = useState("rock_punk");
  const [tempo, setTempo] = useState(140);
  const [duration, setDuration] = useState(64);
  
  // Sampling parameters
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  
  // State
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentArrangement, setCurrentArrangement] = useState<ArrangementResult | null>(null);
  
  // Persistent storage
  const [savedArrangements, setSavedArrangements] = useKV("arrangement-history", []);
  
  // Style definitions
  const styles = [
    { value: "rock_punk", label: "Rock/Punk", description: "High energy, driving rhythms" },
    { value: "rnb_ballad", label: "R&B Ballad", description: "Smooth, emotional progressions" },
    { value: "country_pop", label: "Country Pop", description: "Accessible, narrative structures" }
  ];
  
  const sectionColors = {
    INTRO: "bg-blue-100 text-blue-800 border-blue-300",
    VERSE: "bg-green-100 text-green-800 border-green-300", 
    CHORUS: "bg-purple-100 text-purple-800 border-purple-300",
    BRIDGE: "bg-orange-100 text-orange-800 border-orange-300",
    OUTRO: "bg-gray-100 text-gray-800 border-gray-300"
  };

  // Mock transformer inference (simulates the actual model)
  const generateArrangement = async () => {
    setIsGenerating(true);
    setProgress(0);
    
    try {
      // Simulate model loading and inference
      const steps = ["Loading model...", "Encoding conditions...", "Generating sequence...", "Decoding tokens..."];
      
      for (let i = 0; i < steps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 800));
        setProgress((i + 1) / steps.length * 100);
      }
      
      // Generate mock arrangement based on style and parameters
      const mockArrangement = generateMockArrangement(style, tempo, duration, temperature);
      
      setCurrentArrangement(mockArrangement);
      toast.success("Arrangement generated successfully!");
      
    } catch (error) {
      toast.error("Failed to generate arrangement");
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };
  
  // Mock arrangement generation with style-specific patterns
  const generateMockArrangement = (style: string, tempo: number, duration: number, temp: number): ArrangementResult => {
    const sections: Section[] = [];
    let currentBar = 0;
    
    // Style-specific patterns
    const patterns = {
      rock_punk: [
        { type: "INTRO", length: 4 },
        { type: "VERSE", length: 16 },
        { type: "CHORUS", length: 16 },
        { type: "VERSE", length: 16 },
        { type: "BRIDGE", length: 8 },
        { type: "OUTRO", length: 4 }
      ],
      rnb_ballad: [
        { type: "INTRO", length: 8 },
        { type: "VERSE", length: 16 },
        { type: "CHORUS", length: 16 },
        { type: "VERSE", length: 16 },
        { type: "CHORUS", length: 16 },
        { type: "BRIDGE", length: 8 },
        { type: "OUTRO", length: 4 }
      ],
      country_pop: [
        { type: "INTRO", length: 4 },
        { type: "VERSE", length: 16 },
        { type: "CHORUS", length: 16 },
        { type: "VERSE", length: 16 },
        { type: "CHORUS", length: 16 }
      ]
    };
    
    const basePattern = patterns[style as keyof typeof patterns] || patterns.rock_punk;
    
    // Add some randomness based on temperature
    const variation = Math.random() * temp;
    
    for (const section of basePattern) {
      // Add some variation to section lengths
      const lengthVariation = Math.floor((Math.random() - 0.5) * variation * 8);
      const adjustedLength = Math.max(2, section.length + lengthVariation);
      
      if (currentBar + adjustedLength <= duration) {
        sections.push({
          type: section.type,
          start_bar: currentBar,
          length_bars: adjustedLength
        });
        currentBar += adjustedLength;
      }
    }
    
    // Fill remaining duration if needed
    if (currentBar < duration) {
      const remaining = duration - currentBar;
      if (remaining >= 4) {
        sections.push({
          type: "OUTRO",
          start_bar: currentBar,
          length_bars: remaining
        });
      }
    }
    
    return {
      style,
      tempo,
      duration_bars: duration,
      sections,
      generation_params: {
        temperature,
        top_k: topK,
        top_p: topP
      }
    };
  };
  
  const saveArrangement = () => {
    if (!currentArrangement) return;
    
    const newArrangement = {
      ...currentArrangement,
      id: Date.now(),
      created_at: new Date().toISOString()
    };
    
    setSavedArrangements((prev: any[]) => [newArrangement, ...prev.slice(0, 9)]); // Keep last 10
    toast.success("Arrangement saved to history");
  };
  
  const downloadArrangement = () => {
    if (!currentArrangement) return;
    
    const dataStr = JSON.stringify(currentArrangement, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `arrangement_${style}_${tempo}bpm_${duration}bars.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    toast.success("Arrangement downloaded");
  };
  
  return (
    <div className="space-y-6">
      {/* Main Generation Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-accent" />
            Arrangement Transformer
          </CardTitle>
          <CardDescription>
            Generate song arrangements using a transformer decoder with style conditioning, teacher forcing, and coverage penalty
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Input Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Style Selection */}
            <div className="space-y-2">
              <Label htmlFor="style">Style</Label>
              <Select value={style} onValueChange={setStyle}>
                <SelectTrigger>
                  <SelectValue placeholder="Select style" />
                </SelectTrigger>
                <SelectContent>
                  {styles.map((s) => (
                    <SelectItem key={s.value} value={s.value}>
                      <div>
                        <div className="font-medium">{s.label}</div>
                        <div className="text-sm text-muted-foreground">{s.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            {/* Tempo */}
            <div className="space-y-2">
              <Label htmlFor="tempo">Tempo (BPM)</Label>
              <Input
                id="tempo"
                type="number"
                min="60"
                max="200"
                value={tempo}
                onChange={(e) => setTempo(parseInt(e.target.value) || 120)}
              />
            </div>
            
            {/* Duration */}
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (Bars)</Label>
              <Input
                id="duration"
                type="number"
                min="16"
                max="128"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value) || 64)}
              />
            </div>
          </div>
          
          {/* Advanced Parameters */}
          <details className="space-y-4">
            <summary className="flex items-center gap-2 cursor-pointer font-medium">
              <Settings className="w-4 h-4" />
              Advanced Sampling Parameters
            </summary>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pl-6">
              {/* Temperature */}
              <div className="space-y-2">
                <Label>Temperature: {temperature}</Label>
                <Slider
                  value={[temperature]}
                  onValueChange={(value) => setTemperature(value[0])}
                  min={0.1}
                  max={2.0}
                  step={0.1}
                />
                <p className="text-sm text-muted-foreground">Controls randomness (higher = more creative)</p>
              </div>
              
              {/* Top-k */}
              <div className="space-y-2">
                <Label>Top-k: {topK}</Label>
                <Slider
                  value={[topK]}
                  onValueChange={(value) => setTopK(value[0])}
                  min={1}
                  max={100}
                  step={1}
                />
                <p className="text-sm text-muted-foreground">Number of top tokens to consider</p>
              </div>
              
              {/* Top-p */}
              <div className="space-y-2">
                <Label>Top-p: {topP}</Label>
                <Slider
                  value={[topP]}
                  onValueChange={(value) => setTopP(value[0])}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                />
                <p className="text-sm text-muted-foreground">Cumulative probability threshold</p>
              </div>
            </div>
          </details>
          
          {/* Generation Controls */}
          <div className="flex flex-wrap gap-3">
            <Button 
              onClick={generateArrangement} 
              disabled={isGenerating}
              className="flex items-center gap-2"
            >
              {isGenerating ? (
                <>
                  <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Shuffle className="w-4 h-4" />
                  Generate Arrangement
                </>
              )}
            </Button>
            
            {currentArrangement && (
              <>
                <Button variant="outline" onClick={saveArrangement}>
                  Save to History
                </Button>
                <Button variant="outline" onClick={downloadArrangement}>
                  <Download className="w-4 h-4 mr-2" />
                  Download JSON
                </Button>
              </>
            )}
          </div>
          
          {/* Progress */}
          {isGenerating && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <p className="text-sm text-muted-foreground text-center">
                Generating arrangement with transformer decoder...
              </p>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Generated Arrangement Display */}
      {currentArrangement && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-accent" />
              Generated Arrangement
            </CardTitle>
            <CardDescription>
              {currentArrangement.style.replace('_', ' ')} • {currentArrangement.tempo} BPM • {currentArrangement.sections.length} sections
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Arrangement Timeline */}
            <div className="space-y-3">
              {currentArrangement.sections.map((section, index) => {
                const endBar = section.start_bar + section.length_bars;
                const colorClass = sectionColors[section.type as keyof typeof sectionColors] || sectionColors.INTRO;
                
                return (
                  <div key={index} className="flex items-center gap-4">
                    <div className="w-8 text-sm text-muted-foreground text-right">
                      {index + 1}
                    </div>
                    <Badge variant="outline" className={`${colorClass} min-w-[80px] justify-center`}>
                      {section.type}
                    </Badge>
                    <div className="flex-1 flex items-center gap-2">
                      <span className="text-sm">
                        Bars {section.start_bar}-{endBar} ({section.length_bars} bars)
                      </span>
                      <div className="flex-1 bg-muted rounded-full h-2 relative">
                        <div 
                          className="bg-accent rounded-full h-2 transition-all duration-300"
                          style={{ 
                            width: `${(section.length_bars / currentArrangement.duration_bars) * 100}%`,
                            marginLeft: `${(section.start_bar / currentArrangement.duration_bars) * 100}%`
                          }}
                        />
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {Math.round((section.length_bars * 4 * 60) / currentArrangement.tempo)}s
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">
                  {currentArrangement.sections.length}
                </div>
                <div className="text-sm text-muted-foreground">Sections</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">
                  {currentArrangement.duration_bars}
                </div>
                <div className="text-sm text-muted-foreground">Bars</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">
                  {Math.round((currentArrangement.duration_bars * 4 * 60) / currentArrangement.tempo)}s
                </div>
                <div className="text-sm text-muted-foreground">Duration</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">
                  {currentArrangement.tempo}
                </div>
                <div className="text-sm text-muted-foreground">BPM</div>
              </div>
            </div>
            
            {/* Generation Parameters */}
            {currentArrangement.generation_params && (
              <div className="pt-4 border-t">
                <h4 className="text-sm font-medium mb-2">Generation Parameters</h4>
                <div className="flex gap-4 text-sm text-muted-foreground">
                  <span>Temperature: {currentArrangement.generation_params.temperature}</span>
                  <span>Top-k: {currentArrangement.generation_params.top_k}</span>
                  <span>Top-p: {currentArrangement.generation_params.top_p}</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
      
      {/* History */}
      {savedArrangements.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Arrangements</CardTitle>
            <CardDescription>
              Previously generated arrangements (last 10)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {savedArrangements.slice(0, 5).map((arrangement: any, index: number) => (
                <div key={arrangement.id} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="text-sm font-medium">
                      {arrangement.style.replace('_', ' ')}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {arrangement.tempo} BPM • {arrangement.duration_bars} bars • {arrangement.sections.length} sections
                    </div>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => setCurrentArrangement(arrangement)}
                  >
                    <Play className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}