import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { useKV } from "@github/spark/hooks";
import { toast } from "sonner";
import { Upload, Music, Brain, Search, Zap, Database, ArrowRight, Waveform } from "@phosphor-icons/react";

interface StyleVector {
  id: string;
  style: string;
  vector: number[];
  confidence: number;
  timestamp: number;
}

interface RetrievedPattern {
  pattern: string;
  similarity: number;
  style: string;
  bars: number;
}

interface BiasSettings {
  enabled: boolean;
  retrievalWeight: number;
  topK: number;
  fusionMethod: "shallow" | "deep" | "interpolation";
}

export function StyleEmbeddingDemo() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<string>("rock_punk");
  const [isProcessing, setIsProcessing] = useState(false);
  const [encodingProgress, setEncodingProgress] = useState(0);
  const [indexingProgress, setIndexingProgress] = useState(0);
  const [currentVector, setCurrentVector] = useState<StyleVector | null>(null);
  const [retrievedPatterns, setRetrievedPatterns] = useState<RetrievedPattern[]>([]);
  const [biasSettings, setBiasSettings] = useKV<BiasSettings>("style-bias-settings", {
    enabled: true,
    retrievalWeight: 0.3,
    topK: 5,
    fusionMethod: "shallow"
  });
  const [generatedSequence, setGeneratedSequence] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
  
  const [styleVectors] = useKV<StyleVector[]>("style-vectors", []);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const styles = [
    { value: "rock_punk", label: "Rock/Punk", color: "bg-red-500" },
    { value: "rnb_ballad", label: "R&B Ballad", color: "bg-purple-500" },
    { value: "country_pop", label: "Country Pop", color: "bg-green-500" }
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file);
      toast.success(`Loaded audio file: ${file.name}`);
    } else {
      toast.error("Please select a valid audio file");
    }
  };

  const simulateAudioEncoding = async (audioFile: File): Promise<StyleVector> => {
    // Simulate audio encoding process
    const totalSteps = 100;
    for (let i = 0; i <= totalSteps; i += 5) {
      setEncodingProgress(i);
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Generate synthetic style vector (512-dimensional)
    const vector = Array.from({ length: 512 }, () => Math.random() * 2 - 1);
    const confidence = Math.random() * 0.3 + 0.7; // 0.7-1.0 range

    return {
      id: `vec_${Date.now()}`,
      style: selectedStyle,
      vector,
      confidence,
      timestamp: Date.now()
    };
  };

  const simulatePatternRetrieval = async (styleVector: StyleVector): Promise<RetrievedPattern[]> => {
    // Simulate FAISS index lookup
    const patterns = [
      "KICK BAR_1 POS_1 | SNARE BAR_1 POS_3 | BASS_PICK C2 BAR_1 POS_1",
      "CHORD C BAR_1 | CHORD Am BAR_2 | CHORD F BAR_3 | CHORD G BAR_4",
      "NOTE_ON E4 VEL_80 DUR_8 BAR_1 POS_1 | NOTE_ON G4 VEL_75 DUR_4 BAR_1 POS_2",
      "ACOUSTIC_STRUM Em BAR_1 POS_1 | ACOUSTIC_STRUM Am BAR_1 POS_3",
      "PIANO C4 BAR_1 POS_1 | PIANO E4 BAR_1 POS_2 | PIANO G4 BAR_1 POS_3"
    ];

    return patterns.map((pattern, idx) => ({
      pattern,
      similarity: Math.random() * 0.4 + 0.6, // 0.6-1.0 similarity
      style: styleVector.style,
      bars: Math.floor(Math.random() * 4) + 1
    })).sort((a, b) => b.similarity - a.similarity).slice(0, biasSettings.topK);
  };

  const handleEncodeAudio = async () => {
    if (!audioFile) {
      toast.error("Please select an audio file first");
      return;
    }

    setIsProcessing(true);
    setEncodingProgress(0);

    try {
      const styleVector = await simulateAudioEncoding(audioFile);
      setCurrentVector(styleVector);
      
      // Simulate indexing
      const totalIndexSteps = 100;
      for (let i = 0; i <= totalIndexSteps; i += 10) {
        setIndexingProgress(i);
        await new Promise(resolve => setTimeout(resolve, 30));
      }

      toast.success("Audio encoded and indexed successfully!");
    } catch (error) {
      toast.error("Failed to encode audio");
    } finally {
      setIsProcessing(false);
      setEncodingProgress(0);
      setIndexingProgress(0);
    }
  };

  const handleRetrievePatterns = async () => {
    if (!currentVector) {
      toast.error("Please encode audio first");
      return;
    }

    setIsProcessing(true);
    
    try {
      const patterns = await simulatePatternRetrieval(currentVector);
      setRetrievedPatterns(patterns);
      toast.success(`Retrieved ${patterns.length} similar patterns`);
    } catch (error) {
      toast.error("Failed to retrieve patterns");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerateWithBias = async () => {
    if (retrievedPatterns.length === 0) {
      toast.error("Please retrieve patterns first");
      return;
    }

    setIsGenerating(true);
    
    try {
      // Simulate biased generation
      const baseSequence = "STYLE=rock_punk TEMPO=140 KEY=C SECTION=VERSE BAR_1";
      const biasedTokens = retrievedPatterns
        .slice(0, biasSettings.topK)
        .map(p => p.pattern.split(' ').slice(0, 3).join(' '))
        .join(' | ');
        
      const fusionMethod = biasSettings.enabled ? 
        ` [FUSION=${biasSettings.fusionMethod} WEIGHT=${biasSettings.retrievalWeight}]` : '';
        
      const generated = `${baseSequence} ${biasedTokens}${fusionMethod}`;
      
      // Simulate generation delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setGeneratedSequence(generated);
      toast.success("Sequence generated with retrieval bias!");
    } catch (error) {
      toast.error("Failed to generate sequence");
    } finally {
      setIsGenerating(false);
    }
  };

  const updateBiasSettings = (key: keyof BiasSettings, value: any) => {
    setBiasSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Audio Encoder */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Waveform className="w-5 h-5 text-blue-500" />
              Audio Style Encoder
            </CardTitle>
            <CardDescription>
              Encode 10-second audio clips into style vectors using log-mel spectrograms
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="audio-upload">Audio File (10s clip)</Label>
              <div className="flex gap-2">
                <Input
                  id="audio-upload"
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  ref={fileInputRef}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  {audioFile ? audioFile.name : "Choose Audio File"}
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Target Style</Label>
              <Select value={selectedStyle} onValueChange={setSelectedStyle}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {styles.map(style => (
                    <SelectItem key={style.value} value={style.value}>
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${style.color}`} />
                        {style.label}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {isProcessing && (
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Encoding Audio</span>
                    <span>{encodingProgress}%</span>
                  </div>
                  <Progress value={encodingProgress} className="h-2" />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Building Index</span>
                    <span>{indexingProgress}%</span>
                  </div>
                  <Progress value={indexingProgress} className="h-2" />
                </div>
              </div>
            )}

            <Button 
              onClick={handleEncodeAudio} 
              disabled={!audioFile || isProcessing}
              className="w-full"
            >
              <Brain className="w-4 h-4 mr-2" />
              {isProcessing ? "Encoding..." : "Encode & Index"}
            </Button>

            {currentVector && (
              <div className="p-3 bg-muted rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Style Vector</span>
                  <Badge variant="secondary">{currentVector.confidence.toFixed(3)} confidence</Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  512-dimensional embedding • {currentVector.style}
                </div>
                <div className="text-xs font-mono bg-background p-2 rounded text-muted-foreground">
                  [{currentVector.vector.slice(0, 8).map(v => v.toFixed(3)).join(', ')}...]
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Pattern Retrieval */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-green-500" />
              FAISS Pattern Retrieval
            </CardTitle>
            <CardDescription>
              Retrieve similar musical patterns from indexed reference bars
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              onClick={handleRetrievePatterns}
              disabled={!currentVector || isProcessing}
              className="w-full"
            >
              <Search className="w-4 h-4 mr-2" />
              Retrieve Patterns
            </Button>

            {retrievedPatterns.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-sm font-medium">Retrieved Patterns</h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {retrievedPatterns.map((pattern, idx) => (
                    <div key={idx} className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="outline">{pattern.similarity.toFixed(3)} similarity</Badge>
                        <span className="text-xs text-muted-foreground">{pattern.bars} bars</span>
                      </div>
                      <div className="text-xs font-mono bg-background p-2 rounded">
                        {pattern.pattern}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Retrieval Fusion Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            Retrieval Fusion Settings
          </CardTitle>
          <CardDescription>
            Configure how retrieved patterns bias token generation during decoding
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="settings" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="settings">Bias Settings</TabsTrigger>
              <TabsTrigger value="generation">Generation</TabsTrigger>
            </TabsList>
            
            <TabsContent value="settings" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="bias-enabled">Enable Retrieval Bias</Label>
                    <Switch
                      id="bias-enabled"
                      checked={biasSettings.enabled}
                      onCheckedChange={(checked) => updateBiasSettings('enabled', checked)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Retrieval Weight ({biasSettings.retrievalWeight})</Label>
                    <Slider
                      value={[biasSettings.retrievalWeight]}
                      onValueChange={([value]) => updateBiasSettings('retrievalWeight', value)}
                      max={1}
                      min={0}
                      step={0.1}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Top-K Patterns ({biasSettings.topK})</Label>
                    <Slider
                      value={[biasSettings.topK]}
                      onValueChange={([value]) => updateBiasSettings('topK', value)}
                      max={10}
                      min={1}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Fusion Method</Label>
                    <Select 
                      value={biasSettings.fusionMethod} 
                      onValueChange={(value: "shallow" | "deep" | "interpolation") => 
                        updateBiasSettings('fusionMethod', value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="shallow">Shallow Fusion</SelectItem>
                        <SelectItem value="deep">Deep Fusion</SelectItem>
                        <SelectItem value="interpolation">Interpolation</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="p-3 bg-muted rounded-lg">
                    <h4 className="text-sm font-medium mb-2">Fusion Methods</h4>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <div><strong>Shallow:</strong> Bias output logits only</div>
                      <div><strong>Deep:</strong> Bias at multiple layers</div>
                      <div><strong>Interpolation:</strong> Weighted average of distributions</div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="generation" className="space-y-4">
              <Button 
                onClick={handleGenerateWithBias}
                disabled={retrievedPatterns.length === 0 || isGenerating}
                className="w-full"
                size="lg"
              >
                <ArrowRight className="w-4 h-4 mr-2" />
                {isGenerating ? "Generating..." : "Generate with Retrieval Bias"}
              </Button>

              {generatedSequence && (
                <div className="space-y-2">
                  <Label>Generated Sequence</Label>
                  <Textarea
                    value={generatedSequence}
                    readOnly
                    className="font-mono text-sm min-h-24"
                  />
                  <div className="text-xs text-muted-foreground">
                    Sequence generated using {biasSettings.fusionMethod} fusion with {biasSettings.retrievalWeight} weight
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}