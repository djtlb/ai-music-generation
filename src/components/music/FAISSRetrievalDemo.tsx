import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import { useKV } from '@github/spark/hooks';
import { 
  Database, 
  Search, 
  ArrowRight, 
  FileText, 
  Play, 
  Settings,
  TrendUp,
  Target,
  Brain,
  TreeStructure
} from '@phosphor-icons/react';

interface FAISSIndexInfo {
  parentGenre: string;
  totalPatterns: number;
  childGenres: string[];
  buildTime: string;
  embeddingDim: number;
}

interface RetrievalResult {
  pattern: {
    tokens: string[];
    parentGenre: string;
    childGenre?: string;
    weight: number;
    similarity: number;
  };
  biasApplied: boolean;
  fusionWeight: number;
}

export function FAISSRetrievalDemo() {
  const [indices, setIndices] = useKV<FAISSIndexInfo[]>('faiss-indices', []);
  const [buildProgress, setBuildProgress] = useState(0);
  const [isBuilding, setIsBuilding] = useState(false);
  const [retrievalResults, setRetrievalResults] = useState<RetrievalResult[]>([]);
  
  // Query parameters
  const [queryTokens, setQueryTokens] = useState('STYLE=pop TEMPO=120 CHORD=C NOTE_ON 60 VEL=80');
  const [familyIndex, setFamilyIndex] = useState('pop');
  const [childBias, setChildBias] = useState(0.3);
  const [childGenre, setChildGenre] = useState('dance_pop');
  const [fusionWeight, setFusionWeight] = useState(0.1);
  const [topK, setTopK] = useState(5);

  const parentGenres = [
    'pop', 'rock', 'rnb_soul', 'country', 'hiphop_rap', 
    'dance_edm', 'jazz_influenced', 'latin', 'afro',
    'reggae_dancehall', 'kpop_jpop', 'singer_songwriter', 'christian_gospel'
  ];

  const childGenreMap: Record<string, string[]> = {
    pop: ['dance_pop', 'pop_rock', 'synth_pop', 'indie_pop'],
    rock: ['punk', 'alt_rock', 'indie_rock', 'prog_rock'],
    rnb_soul: ['neo_soul', 'contemporary_rnb', 'funk'],
    country: ['country_pop', 'country_rock', 'bluegrass'],
    hiphop_rap: ['trap', 'boom_bap', 'conscious_rap', 'mumble_rap'],
    dance_edm: ['house', 'techno', 'dubstep', 'trance']
  };

  const simulateBuildProcess = async () => {
    setIsBuilding(true);
    setBuildProgress(0);

    const steps = [
      'Loading tokenized MIDI patterns...',
      'Creating pattern embeddings...',
      'Building parent FAISS indices...',
      'Registering child patterns...',
      'Applying hierarchical weights...',
      'Saving indices to disk...'
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setBuildProgress(((i + 1) / steps.length) * 100);
      toast.info(steps[i]);
    }

    // Simulate creating indices
    const newIndices: FAISSIndexInfo[] = parentGenres.slice(0, 3).map(genre => ({
      parentGenre: genre,
      totalPatterns: Math.floor(Math.random() * 500) + 100,
      childGenres: childGenreMap[genre] || [],
      buildTime: new Date().toISOString(),
      embeddingDim: 512
    }));

    setIndices(currentIndices => [...currentIndices, ...newIndices]);
    setIsBuilding(false);
    toast.success('FAISS indices built successfully!');
  };

  const simulateRetrieval = async () => {
    const tokens = queryTokens.split(' ');
    
    toast.info('Searching for similar patterns...');
    
    await new Promise(resolve => setTimeout(resolve, 800));

    // Simulate retrieval results
    const mockResults: RetrievalResult[] = [
      {
        pattern: {
          tokens: ['STYLE=pop', 'TEMPO=120', 'CHORD=C', 'NOTE_ON', '60', 'VEL=85'],
          parentGenre: familyIndex,
          similarity: 0.95,
          weight: 1.0
        },
        biasApplied: false,
        fusionWeight: fusionWeight
      },
      {
        pattern: {
          tokens: ['STYLE=dance_pop', 'TEMPO=128', 'CHORD=C', 'NOTE_ON', '60', 'VEL=90'],
          parentGenre: familyIndex,
          childGenre: childGenre,
          similarity: 0.87,
          weight: 1.5
        },
        biasApplied: childBias > 0,
        fusionWeight: fusionWeight * (1 + childBias)
      },
      {
        pattern: {
          tokens: ['STYLE=pop', 'TEMPO=115', 'CHORD=F', 'NOTE_ON', '65', 'VEL=75'],
          parentGenre: familyIndex,
          similarity: 0.82,
          weight: 1.0
        },
        biasApplied: false,
        fusionWeight: fusionWeight
      },
      {
        pattern: {
          tokens: ['STYLE=dance_pop', 'TEMPO=126', 'CHORD=Am', 'NOTE_ON', '57', 'VEL=88'],
          parentGenre: familyIndex,
          childGenre: childGenre,
          similarity: 0.79,
          weight: 1.5
        },
        biasApplied: childBias > 0,
        fusionWeight: fusionWeight * (1 + childBias)
      },
      {
        pattern: {
          tokens: ['STYLE=pop', 'TEMPO=118', 'CHORD=G', 'NOTE_ON', '67', 'VEL=78'],
          parentGenre: familyIndex,
          similarity: 0.74,
          weight: 1.0
        },
        biasApplied: false,
        fusionWeight: fusionWeight
      }
    ].slice(0, topK);

    setRetrievalResults(mockResults);
    toast.success(`Retrieved ${mockResults.length} similar patterns with bias applied`);
  };

  const clearIndices = () => {
    setIndices([]);
    setRetrievalResults([]);
    toast.info('Cleared all indices');
  };

  return (
    <div className="space-y-6">
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="build">Build Indices</TabsTrigger>
          <TabsTrigger value="query">Query & Retrieval</TabsTrigger>
          <TabsTrigger value="results">Results Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TreeStructure className="w-5 h-5 text-primary" />
                Hierarchical FAISS Index System
              </CardTitle>
              <CardDescription>
                Build and query FAISS indices for parent-child style pattern retrieval with bias weighting
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Database className="w-4 h-4 text-primary" />
                    <span className="font-medium">Parent Indices</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    FAISS indices built from tokenized MIDI patterns in each parent genre&apos;s refs_midi directory
                  </p>
                </div>
                
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-4 h-4 text-accent" />
                    <span className="font-medium">Child Bias</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Child patterns registered with higher fusion weights to bias generation toward specific sub-genres
                  </p>
                </div>
                
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-secondary" />
                    <span className="font-medium">Retrieval Fusion</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    N-gram matching and similarity search combined with logit fusion during token generation
                  </p>
                </div>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-medium mb-2">Architecture Flow</h4>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span className="px-2 py-1 bg-primary/10 rounded">refs_midi/*.tokens</span>
                  <ArrowRight className="w-3 h-3" />
                  <span className="px-2 py-1 bg-primary/10 rounded">Pattern Embeddings</span>
                  <ArrowRight className="w-3 h-3" />
                  <span className="px-2 py-1 bg-primary/10 rounded">FAISS Index</span>
                  <ArrowRight className="w-3 h-3" />
                  <span className="px-2 py-1 bg-accent/10 rounded">Child Bias</span>
                  <ArrowRight className="w-3 h-3" />
                  <span className="px-2 py-1 bg-secondary/10 rounded">Retrieval Fusion</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="build" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5 text-primary" />
                Build FAISS Indices
              </CardTitle>
              <CardDescription>
                Create hierarchical indices from tokenized MIDI patterns with parent-child relationships
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {isBuilding && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Building indices...</span>
                    <span className="text-sm text-muted-foreground">{Math.round(buildProgress)}%</span>
                  </div>
                  <Progress value={buildProgress} className="w-full" />
                </div>
              )}

              <div className="flex gap-2">
                <Button 
                  onClick={simulateBuildProcess} 
                  disabled={isBuilding}
                  className="flex items-center gap-2"
                >
                  <Database className="w-4 h-4" />
                  Build Hierarchical Indices
                </Button>
                
                <Button 
                  variant="outline" 
                  onClick={clearIndices}
                  disabled={isBuilding}
                >
                  Clear Indices
                </Button>
              </div>

              {indices.length > 0 && (
                <div className="mt-6">
                  <h4 className="font-medium mb-3">Built Indices</h4>
                  <div className="grid gap-3">
                    {indices.map((index, idx) => (
                      <div key={idx} className="p-3 border rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">{index.parentGenre}</Badge>
                            <span className="text-sm text-muted-foreground">
                              {index.totalPatterns} patterns
                            </span>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {index.embeddingDim}D embeddings
                          </span>
                        </div>
                        
                        {index.childGenres.length > 0 && (
                          <div className="flex gap-1 flex-wrap">
                            <span className="text-xs text-muted-foreground mr-2">Children:</span>
                            {index.childGenres.map(child => (
                              <Badge key={child} variant="secondary" className="text-xs">
                                {child}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="query" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5 text-primary" />
                Query Configuration
              </CardTitle>
              <CardDescription>
                Configure retrieval parameters and test pattern matching with hierarchical bias
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="query-tokens">Query Tokens</Label>
                  <Textarea
                    id="query-tokens"
                    value={queryTokens}
                    onChange={(e) => setQueryTokens(e.target.value)}
                    placeholder="STYLE=pop TEMPO=120 CHORD=C NOTE_ON 60 VEL=80"
                    rows={3}
                  />
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="family-index">Parent Genre (family_index)</Label>
                    <Select value={familyIndex} onValueChange={setFamilyIndex}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {parentGenres.map(genre => (
                          <SelectItem key={genre} value={genre}>
                            {genre.replace('_', ' ')}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="child-genre">Child Genre</Label>
                    <Select value={childGenre} onValueChange={setChildGenre}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {(childGenreMap[familyIndex] || []).map(child => (
                          <SelectItem key={child} value={child}>
                            {child.replace('_', ' ')}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="child-bias">Child Bias</Label>
                  <Input
                    id="child-bias"
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={childBias}
                    onChange={(e) => setChildBias(parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="fusion-weight">Fusion Weight</Label>
                  <Input
                    id="fusion-weight"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={fusionWeight}
                    onChange={(e) => setFusionWeight(parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="top-k">Top K Results</Label>
                  <Input
                    id="top-k"
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                  />
                </div>

                <div className="flex items-end">
                  <Button 
                    onClick={simulateRetrieval}
                    disabled={indices.length === 0}
                    className="w-full flex items-center gap-2"
                  >
                    <Search className="w-4 h-4" />
                    Query Index
                  </Button>
                </div>
              </div>

              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  <strong>Command:</strong> --family_index {familyIndex} --child_bias {childBias}
                  {childGenre && ` --child_genre ${childGenre}`}
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendUp className="w-5 h-5 text-primary" />
                Retrieval Results
              </CardTitle>
              <CardDescription>
                Analysis of retrieved patterns with parent-child bias effects
              </CardDescription>
            </CardHeader>
            <CardContent>
              {retrievalResults.length > 0 ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="p-3 bg-primary/5 rounded-lg">
                      <div className="text-sm font-medium text-primary">Total Results</div>
                      <div className="text-2xl font-bold">{retrievalResults.length}</div>
                    </div>
                    
                    <div className="p-3 bg-accent/5 rounded-lg">
                      <div className="text-sm font-medium text-accent">Child Biased</div>
                      <div className="text-2xl font-bold">
                        {retrievalResults.filter(r => r.biasApplied).length}
                      </div>
                    </div>
                    
                    <div className="p-3 bg-secondary/5 rounded-lg">
                      <div className="text-sm font-medium text-secondary">Avg Similarity</div>
                      <div className="text-2xl font-bold">
                        {(retrievalResults.reduce((sum, r) => sum + r.pattern.similarity, 0) / retrievalResults.length).toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {retrievalResults.map((result, idx) => (
                      <div 
                        key={idx} 
                        className={`p-4 border rounded-lg ${result.biasApplied ? 'border-accent/50 bg-accent/5' : 'border-border'}`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Badge variant={result.pattern.childGenre ? "default" : "outline"}>
                              {result.pattern.childGenre || result.pattern.parentGenre}
                            </Badge>
                            
                            {result.biasApplied && (
                              <Badge variant="secondary" className="text-xs">
                                Biased +{Math.round((result.pattern.weight - 1) * 100)}%
                              </Badge>
                            )}
                            
                            <span className="text-sm text-muted-foreground">
                              Similarity: {result.pattern.similarity.toFixed(3)}
                            </span>
                          </div>
                          
                          <div className="text-right">
                            <div className="text-xs text-muted-foreground">
                              Fusion Weight: {result.fusionWeight.toFixed(3)}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex gap-1 flex-wrap">
                          {result.pattern.tokens.map((token, tokenIdx) => (
                            <span 
                              key={tokenIdx}
                              className="px-2 py-1 bg-muted text-xs rounded font-mono"
                            >
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No retrieval results yet. Configure and run a query to see pattern matches.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}