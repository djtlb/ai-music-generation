import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { MidiTokenizer, MultiTrackMidi } from '../models/tokenizer';
import TokenizerTester, { testFixtures, TestResult } from '../models/tokenizer.test';
import { smokeTestResults } from '../notebooks/tokenizer_smoke';
import { FileCode, Play, CheckCircle, XCircle, Info, Beaker } from '@phosphor-icons/react';
import { toast } from 'sonner';

export function TokenizerDemo() {
  const [tokenizer] = useState(() => new MidiTokenizer());
  const [selectedFixture, setSelectedFixture] = useState<string>('0');
  const [customMidiText, setCustomMidiText] = useState<string>('');
  const [encodedTokens, setEncodedTokens] = useState<number[]>([]);
  const [decodedMidi, setDecodedMidi] = useState<MultiTrackMidi | null>(null);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [activeTab, setActiveTab] = useState('demo');

  const fixtures = testFixtures as MultiTrackMidi[];

  // Initialize with first fixture
  useEffect(() => {
    if (fixtures.length > 0) {
      handleEncode(fixtures[0]);
    }
  }, []);

  const handleEncode = (midi: MultiTrackMidi) => {
    try {
      const tokens = tokenizer.encode(midi);
      setEncodedTokens(tokens);
      
      // Immediately decode to show round-trip
      const decoded = tokenizer.decode(tokens);
      setDecodedMidi(decoded);
      
      toast.success(`Encoded ${Object.values(midi.tracks).flat().length} notes to ${tokens.length} tokens`);
    } catch (error) {
      toast.error(`Encoding failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleFixtureChange = (value: string) => {
    setSelectedFixture(value);
    const fixture = fixtures[parseInt(value)];
    if (fixture) {
      handleEncode(fixture);
    }
  };

  const handleCustomEncode = () => {
    try {
      const midi = JSON.parse(customMidiText) as MultiTrackMidi;
      handleEncode(midi);
      setActiveTab('results');
    } catch (error) {
      toast.error('Invalid JSON format');
    }
  };

  const runComprehensiveTests = async () => {
    setIsRunningTests(true);
    setActiveTab('tests');
    
    try {
      // Simulate async testing for better UX
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const tester = new TokenizerTester();
      const results = tester.runAllTests();
      setTestResults(results);
      
      const passed = results.filter(r => r.passed).length;
      toast.success(`Tests completed: ${passed}/${results.length} passed`);
    } catch (error) {
      toast.error(`Testing failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsRunningTests(false);
    }
  };

  const getTokenPreview = (tokens: number[], maxTokens: number = 20): { preview: string; total: number } => {
    const tokenNames = tokens.slice(0, maxTokens).map(id => tokenizer.getToken(id));
    return {
      preview: tokenNames.join(' • '),
      total: tokens.length
    };
  };

  const formatMidiForDisplay = (midi: MultiTrackMidi | null): string => {
    if (!midi) return '';
    return JSON.stringify(midi, null, 2);
  };

  return (
    <div className=\"space-y-6\">
      {/* Header */}
      <div className=\"flex items-center justify-between\">
        <div>
          <h2 className=\"text-2xl font-bold\">MIDI Tokenizer</h2>
          <p className=\"text-muted-foreground\">
            Convert multi-track MIDI to tokens for ML training with style-aware encoding
          </p>
        </div>
        <div className=\"flex items-center gap-2\">
          <Badge variant=\"outline\">{tokenizer.getVocabSize()} tokens</Badge>
          <Button onClick={runComprehensiveTests} disabled={isRunningTests} variant=\"outline\">
            <Beaker className=\"w-4 h-4 mr-2\" />
            {isRunningTests ? 'Testing...' : 'Run Tests'}
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className=\"grid w-full grid-cols-4\">
          <TabsTrigger value=\"demo\">Demo</TabsTrigger>
          <TabsTrigger value=\"results\">Results</TabsTrigger>
          <TabsTrigger value=\"tests\">Tests</TabsTrigger>
          <TabsTrigger value=\"analysis\">Analysis</TabsTrigger>
        </TabsList>

        {/* Demo Tab */}
        <TabsContent value=\"demo\" className=\"space-y-6\">
          <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
            {/* Input Section */}
            <Card>
              <CardHeader>
                <CardTitle className=\"flex items-center gap-2\">
                  <FileCode className=\"w-5 h-5\" />
                  Input MIDI
                </CardTitle>
                <CardDescription>
                  Select a test fixture or input custom MIDI JSON
                </CardDescription>
              </CardHeader>
              <CardContent className=\"space-y-4\">
                <div>
                  <label className=\"text-sm font-medium mb-2 block\">Test Fixtures</label>
                  <Select value={selectedFixture} onValueChange={handleFixtureChange}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {fixtures.map((fixture, index) => (
                        <SelectItem key={index} value={index.toString()}>
                          {(fixture as any).name || `Fixture ${index + 1}`} - {fixture.style}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className=\"text-sm font-medium mb-2 block\">Custom MIDI JSON</label>
                  <Textarea
                    placeholder=\"Paste MIDI JSON here...\"
                    value={customMidiText}
                    onChange={(e) => setCustomMidiText(e.target.value)}
                    rows={8}
                    className=\"font-mono text-sm\"
                  />
                  <Button onClick={handleCustomEncode} className=\"mt-2\" disabled={!customMidiText.trim()}>
                    <Play className=\"w-4 h-4 mr-2\" />
                    Encode Custom
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Encoding Stats</CardTitle>
                <CardDescription>
                  Current encoding statistics
                </CardDescription>
              </CardHeader>
              <CardContent className=\"space-y-4\">
                {fixtures[parseInt(selectedFixture)] && (
                  <>
                    <div className=\"grid grid-cols-2 gap-4\">
                      <div>
                        <div className=\"text-2xl font-bold text-primary\">
                          {fixtures[parseInt(selectedFixture)].style}
                        </div>
                        <div className=\"text-sm text-muted-foreground\">Style</div>
                      </div>
                      <div>
                        <div className=\"text-2xl font-bold\">
                          {fixtures[parseInt(selectedFixture)].tempo}
                        </div>
                        <div className=\"text-sm text-muted-foreground\">BPM</div>
                      </div>
                      <div>
                        <div className=\"text-2xl font-bold\">
                          {Object.keys(fixtures[parseInt(selectedFixture)].tracks).length}
                        </div>
                        <div className=\"text-sm text-muted-foreground\">Tracks</div>
                      </div>
                      <div>
                        <div className=\"text-2xl font-bold\">
                          {Object.values(fixtures[parseInt(selectedFixture)].tracks).flat().length}
                        </div>
                        <div className=\"text-sm text-muted-foreground\">Notes</div>
                      </div>
                    </div>
                    
                    {encodedTokens.length > 0 && (
                      <div className=\"pt-4 border-t\">
                        <div className=\"text-2xl font-bold text-accent\">
                          {encodedTokens.length}
                        </div>
                        <div className=\"text-sm text-muted-foreground\">Tokens Generated</div>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value=\"results\" className=\"space-y-6\">
          {encodedTokens.length > 0 && (
            <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
              {/* Encoded Tokens */}
              <Card>
                <CardHeader>
                  <CardTitle>Encoded Tokens</CardTitle>
                  <CardDescription>
                    {encodedTokens.length} tokens generated
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <div>
                      <label className=\"text-sm font-medium mb-2 block\">Token IDs (first 20)</label>
                      <div className=\"p-3 bg-muted rounded font-mono text-sm\">
                        {encodedTokens.slice(0, 20).join(', ')}
                        {encodedTokens.length > 20 && ` ... (+${encodedTokens.length - 20} more)`}
                      </div>
                    </div>
                    
                    <div>
                      <label className=\"text-sm font-medium mb-2 block\">Readable Tokens (first 10)</label>
                      <div className=\"p-3 bg-muted rounded text-sm space-y-1\">
                        {encodedTokens.slice(0, 10).map((tokenId, index) => (
                          <div key={index} className=\"flex justify-between\">
                            <span className=\"font-mono\">{tokenId}</span>
                            <span className=\"text-primary\">{tokenizer.getToken(tokenId)}</span>
                          </div>
                        ))}
                        {encodedTokens.length > 10 && (
                          <div className=\"text-muted-foreground\">... {encodedTokens.length - 10} more tokens</div>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Decoded MIDI */}
              <Card>
                <CardHeader>
                  <CardTitle>Decoded MIDI</CardTitle>
                  <CardDescription>
                    Round-trip decoded result
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Textarea
                    value={formatMidiForDisplay(decodedMidi)}
                    readOnly
                    rows={16}
                    className=\"font-mono text-sm\"
                  />
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Tests Tab */}
        <TabsContent value=\"tests\" className=\"space-y-6\">
          <Card>
            <CardHeader>
              <CardTitle>Comprehensive Test Suite</CardTitle>
              <CardDescription>
                Round-trip encoding/decoding validation on all test fixtures
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isRunningTests && (
                <div className=\"space-y-2\">
                  <Progress value={66} />
                  <p className=\"text-sm text-muted-foreground\">Running tests...</p>
                </div>
              )}
              
              {testResults.length > 0 && (
                <div className=\"space-y-4\">
                  <div className=\"flex items-center gap-4\">
                    <div className=\"flex items-center gap-2\">
                      <CheckCircle className=\"w-5 h-5 text-green-500\" />
                      <span>{testResults.filter(r => r.passed).length} Passed</span>
                    </div>
                    <div className=\"flex items-center gap-2\">
                      <XCircle className=\"w-5 h-5 text-red-500\" />
                      <span>{testResults.filter(r => !r.passed).length} Failed</span>
                    </div>
                  </div>
                  
                  <div className=\"space-y-2\">
                    {testResults.map((result, index) => (
                      <div key={index} className=\"flex items-center justify-between p-3 border rounded\">
                        <div className=\"flex items-center gap-3\">
                          {result.passed ? (
                            <CheckCircle className=\"w-4 h-4 text-green-500\" />
                          ) : (
                            <XCircle className=\"w-4 h-4 text-red-500\" />
                          )}
                          <span className=\"font-medium\">{result.testName}</span>
                        </div>
                        
                        <div className=\"flex items-center gap-2 text-sm text-muted-foreground\">
                          {result.originalTokenCount && (
                            <span>{result.originalTokenCount} tokens</span>
                          )}
                          {!result.passed && result.error && (
                            <Badge variant=\"destructive\" className=\"text-xs\">Error</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value=\"analysis\" className=\"space-y-6\">
          <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
            {/* Vocabulary Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Vocabulary Analysis</CardTitle>
                <CardDescription>
                  Token distribution and coverage
                </CardDescription>
              </CardHeader>
              <CardContent className=\"space-y-4\">
                <div className=\"grid grid-cols-2 gap-4\">
                  <div>
                    <div className=\"text-2xl font-bold\">{tokenizer.getVocabSize()}</div>
                    <div className=\"text-sm text-muted-foreground\">Total Tokens</div>
                  </div>
                  <div>
                    <div className=\"text-2xl font-bold\">3</div>
                    <div className=\"text-sm text-muted-foreground\">Styles</div>
                  </div>
                  <div>
                    <div className=\"text-2xl font-bold\">15</div>
                    <div className=\"text-sm text-muted-foreground\">Instruments</div>
                  </div>
                  <div>
                    <div className=\"text-2xl font-bold\">128</div>
                    <div className=\"text-sm text-muted-foreground\">MIDI Pitches</div>
                  </div>
                </div>
                
                <Alert>
                  <Info className=\"w-4 h-4\" />
                  <AlertDescription>
                    Vocabulary includes style tokens (rock_punk, rnb_ballad, country_pop), 
                    instrument roles, temporal positions, and musical elements.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>
                  Encoding/decoding speed analysis
                </CardDescription>
              </CardHeader>
              <CardContent className=\"space-y-4\">
                <div className=\"space-y-3\">
                  {fixtures.map((fixture, index) => {
                    const noteCount = Object.values(fixture.tracks).flat().length;
                    const start = performance.now();
                    const tokens = tokenizer.encode(fixture);
                    const encodeTime = performance.now() - start;
                    
                    return (
                      <div key={index} className=\"flex justify-between items-center p-2 border rounded\">
                        <div>
                          <div className=\"font-medium\">{(fixture as any).name}</div>
                          <div className=\"text-sm text-muted-foreground\">
                            {noteCount} notes → {tokens.length} tokens
                          </div>
                        </div>
                        <div className=\"text-right\">
                          <div className=\"text-sm font-mono\">{encodeTime.toFixed(2)}ms</div>
                          <div className=\"text-xs text-muted-foreground\">encode time</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}