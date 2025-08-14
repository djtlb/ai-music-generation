import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { useKV } from '@github/spark/hooks';
import { 
  Play, 
  Stop, 
  Upload, 
  Download, 
  TrendingUp, 
  Target, 
  Award,
  Brain,
  Settings,
  BarChart3
} from "@phosphor-icons/react";

interface CriticScore {
  hook_strength: number;
  harmonic_stability: number;
  arrangement_contrast: number;
  mix_quality: number;
  style_match: number;
  overall: number;
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface DPOMetrics {
  iteration: number;
  reward_improvement: number;
  kl_divergence: number;
  policy_loss: number;
  win_rate: number;
}

interface AudioClip {
  id: string;
  name: string;
  style: string;
  duration: number;
  scores?: CriticScore;
  preference_rank?: number;
}

export function CriticRewardDemo() {
  const [activeTab, setActiveTab] = useState("critic");
  const [selectedClip, setSelectedClip] = useState<string>("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  
  // Persistent storage for training data and models
  const [trainingData, setTrainingData] = useKV("critic-training-data", []);
  const [criticMetrics, setCriticMetrics] = useKV("critic-metrics", []);
  const [dpoMetrics, setDpoMetrics] = useKV("dpo-metrics", []);
  const [audioClips, setAudioClips] = useKV("audio-clips", []);
  const [preferences, setPreferences] = useKV("preference-pairs", []);

  // Training configuration
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(50);
  const [dpoIterations, setDpoIterations] = useState(100);

  // Mock audio clips for demonstration
  const mockClips: AudioClip[] = [
    {
      id: "clip1",
      name: "Rock Anthem Demo",
      style: "rock_punk",
      duration: 10.2,
      scores: {
        hook_strength: 0.85,
        harmonic_stability: 0.78,
        arrangement_contrast: 0.92,
        mix_quality: 0.76,
        style_match: 0.88,
        overall: 0.84
      },
      preference_rank: 1
    },
    {
      id: "clip2", 
      name: "R&B Ballad Snippet",
      style: "rnb_ballad",
      duration: 10.0,
      scores: {
        hook_strength: 0.72,
        harmonic_stability: 0.89,
        arrangement_contrast: 0.65,
        mix_quality: 0.91,
        style_match: 0.83,
        overall: 0.80
      },
      preference_rank: 2
    },
    {
      id: "clip3",
      name: "Country Pop Hook",
      style: "country_pop", 
      duration: 9.8,
      scores: {
        hook_strength: 0.94,
        harmonic_stability: 0.71,
        arrangement_contrast: 0.68,
        mix_quality: 0.82,
        style_match: 0.79,
        overall: 0.79
      },
      preference_rank: 3
    }
  ];

  const simulateTraining = async () => {
    setIsTraining(true);
    
    // Simulate training epochs
    const newMetrics: TrainingMetrics[] = [];
    
    for (let epoch = 1; epoch <= epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const metrics: TrainingMetrics = {
        epoch,
        loss: Math.max(0.1, 2.0 - (epoch * 0.035) + (Math.random() - 0.5) * 0.1),
        accuracy: Math.min(0.95, 0.3 + (epoch * 0.012) + (Math.random() - 0.5) * 0.02),
        val_loss: Math.max(0.15, 2.2 - (epoch * 0.032) + (Math.random() - 0.5) * 0.15),
        val_accuracy: Math.min(0.92, 0.25 + (epoch * 0.011) + (Math.random() - 0.5) * 0.03)
      };
      
      newMetrics.push(metrics);
      setCriticMetrics(newMetrics);
    }
    
    setIsTraining(false);
    toast.success("Critic model training completed!");
  };

  const simulateDPOFinetuning = async () => {
    setIsTraining(true);
    
    const newDPOMetrics: DPOMetrics[] = [];
    
    for (let iteration = 1; iteration <= dpoIterations; iteration++) {
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const metrics: DPOMetrics = {
        iteration,
        reward_improvement: Math.max(0, (iteration * 0.008) + (Math.random() - 0.5) * 0.02),
        kl_divergence: Math.max(0.01, 0.5 - (iteration * 0.003) + (Math.random() - 0.5) * 0.05),
        policy_loss: Math.max(0.1, 1.5 - (iteration * 0.012) + (Math.random() - 0.5) * 0.08),
        win_rate: Math.min(0.85, 0.5 + (iteration * 0.003) + (Math.random() - 0.5) * 0.02)
      };
      
      newDPOMetrics.push(metrics);
      setDpoMetrics(newDPOMetrics);
    }
    
    setIsTraining(false);
    toast.success("DPO finetuning completed! Policy improved significantly.");
  };

  const evaluateClip = async (clipId: string) => {
    setIsEvaluating(true);
    setSelectedClip(clipId);
    
    // Simulate evaluation
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const clip = mockClips.find(c => c.id === clipId);
    if (clip?.scores) {
      toast.success(`Evaluation complete! Overall score: ${(clip.scores.overall * 100).toFixed(1)}%`);
    }
    
    setIsEvaluating(false);
  };

  const renderScoreCard = (label: string, score: number, description: string) => (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold">{(score * 100).toFixed(0)}%</span>
            <Badge variant={score > 0.8 ? "default" : score > 0.6 ? "secondary" : "destructive"}>
              {score > 0.8 ? "Excellent" : score > 0.6 ? "Good" : "Needs Work"}
            </Badge>
          </div>
          <Progress value={score * 100} className="h-2" />
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Training Status</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isTraining ? "Training..." : "Ready"}
            </div>
            <p className="text-xs text-muted-foreground">
              {criticMetrics.length > 0 ? `${criticMetrics.length} epochs completed` : "No training data"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Performance</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {criticMetrics.length > 0 ? 
                `${(criticMetrics[criticMetrics.length - 1]?.val_accuracy * 100).toFixed(1)}%` : 
                "N/A"
              }
            </div>
            <p className="text-xs text-muted-foreground">
              Validation accuracy
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">DPO Win Rate</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dpoMetrics.length > 0 ? 
                `${(dpoMetrics[dpoMetrics.length - 1]?.win_rate * 100).toFixed(1)}%` : 
                "N/A"
              }
            </div>
            <p className="text-xs text-muted-foreground">
              Preference alignment score
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="critic">Critic Model</TabsTrigger>
          <TabsTrigger value="dpo">DPO Finetuning</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="critic" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Critic Training Configuration</CardTitle>
              <CardDescription>
                Configure and train the reward model that scores audio clips on multiple quality dimensions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="batch-size">Batch Size</Label>
                  <Input
                    id="batch-size"
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="learning-rate">Learning Rate</Label>
                  <Input
                    id="learning-rate"
                    type="number"
                    step="0.0001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="epochs">Epochs</Label>
                  <Input
                    id="epochs"
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                  />
                </div>
              </div>
              
              <div className="flex gap-2">
                <Button 
                  onClick={simulateTraining}
                  disabled={isTraining}
                  className="flex-1"
                >
                  {isTraining ? "Training..." : "Start Critic Training"}
                </Button>
                <Button variant="outline" size="icon">
                  <Upload className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon">
                  <Download className="h-4 w-4" />
                </Button>
              </div>

              {criticMetrics.length > 0 && (
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <Label>Training Loss</Label>
                    <div className="text-2xl font-bold">
                      {criticMetrics[criticMetrics.length - 1]?.loss.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <Label>Validation Accuracy</Label>
                    <div className="text-2xl font-bold">
                      {(criticMetrics[criticMetrics.length - 1]?.val_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dpo" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>DPO Finetuning</CardTitle>
              <CardDescription>
                Use Direct Preference Optimization to align the symbolic generator with human preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="dpo-iterations">DPO Iterations</Label>
                  <Input
                    id="dpo-iterations"
                    type="number"
                    value={dpoIterations}
                    onChange={(e) => setDpoIterations(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="policy-lr">Policy Learning Rate</Label>
                  <Input
                    id="policy-lr"
                    type="number"
                    step="0.00001"
                    value={0.00005}
                    readOnly
                  />
                </div>
              </div>

              <Button 
                onClick={simulateDPOFinetuning}
                disabled={isTraining || criticMetrics.length === 0}
                className="w-full"
              >
                {isTraining ? "Finetuning..." : "Start DPO Finetuning"}
              </Button>

              {dpoMetrics.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                  <div>
                    <Label>Reward Improvement</Label>
                    <div className="text-xl font-bold">
                      +{(dpoMetrics[dpoMetrics.length - 1]?.reward_improvement * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <Label>KL Divergence</Label>
                    <div className="text-xl font-bold">
                      {dpoMetrics[dpoMetrics.length - 1]?.kl_divergence.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <Label>Policy Loss</Label>
                    <div className="text-xl font-bold">
                      {dpoMetrics[dpoMetrics.length - 1]?.policy_loss.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <Label>Win Rate</Label>
                    <div className="text-xl font-bold">
                      {(dpoMetrics[dpoMetrics.length - 1]?.win_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evaluation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Audio Clip Evaluation</CardTitle>
              <CardDescription>
                Evaluate audio clips using the trained critic model across multiple quality dimensions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4">
                {mockClips.map((clip) => (
                  <Card key={clip.id} className={selectedClip === clip.id ? "ring-2 ring-primary" : ""}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold">{clip.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            {clip.style} • {clip.duration}s
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">#{clip.preference_rank}</Badge>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIsPlaying(!isPlaying)}
                          >
                            {isPlaying ? <Stop className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => evaluateClip(clip.id)}
                            disabled={isEvaluating}
                          >
                            {isEvaluating && selectedClip === clip.id ? "Evaluating..." : "Evaluate"}
                          </Button>
                        </div>
                      </div>
                      
                      {clip.scores && (
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-4">
                          {renderScoreCard("Hook", clip.scores.hook_strength, "Memorable melodic content")}
                          {renderScoreCard("Harmony", clip.scores.harmonic_stability, "Chord progression quality")}
                          {renderScoreCard("Contrast", clip.scores.arrangement_contrast, "Dynamic variation")}
                          {renderScoreCard("Mix", clip.scores.mix_quality, "LUFS and spectral balance")}
                          {renderScoreCard("Style", clip.scores.style_match, "Genre consistency")}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Critic Training Progress
                </CardTitle>
              </CardHeader>
              <CardContent>
                {criticMetrics.length > 0 ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Final Training Loss</Label>
                        <div className="text-xl font-bold">
                          {criticMetrics[criticMetrics.length - 1]?.loss.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <Label>Final Validation Accuracy</Label>
                        <div className="text-xl font-bold">
                          {(criticMetrics[criticMetrics.length - 1]?.val_accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="h-32 bg-muted rounded flex items-center justify-center">
                      <span className="text-sm text-muted-foreground">Training curves visualization</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No training metrics available. Start critic training first.
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  DPO Optimization Progress  
                </CardTitle>
              </CardHeader>
              <CardContent>
                {dpoMetrics.length > 0 ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Total Reward Gain</Label>
                        <div className="text-xl font-bold text-green-600">
                          +{(dpoMetrics[dpoMetrics.length - 1]?.reward_improvement * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <Label>Final Win Rate</Label>
                        <div className="text-xl font-bold">
                          {(dpoMetrics[dpoMetrics.length - 1]?.win_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="h-32 bg-muted rounded flex items-center justify-center">
                      <span className="text-sm text-muted-foreground">DPO optimization curves</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No DPO metrics available. Complete critic training and start DPO finetuning.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}