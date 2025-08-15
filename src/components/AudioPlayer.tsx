import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { audioSynth } from "@/lib/audio";
import { Play, Pause, Stop, Volume2, VolumeX } from "@phosphor-icons/react";
import { toast } from "sonner";

interface AudioPlayerProps {
  title?: string;
  notes?: Array<{
    pitch: number;
    duration: number;
    velocity?: number;
  }>;
  chords?: string[];
  className?: string;
  disabled?: boolean;
}

export function AudioPlayer({ 
  title = "Audio Preview", 
  notes = [], 
  chords = [], 
  className = "",
  disabled = false 
}: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState([0.3]);
  const [isMuted, setIsMuted] = useState(false);

  const handlePlay = async () => {
    if (isPlaying) {
      audioSynth.stopAll();
      setIsPlaying(false);
      return;
    }

    if (notes.length === 0 && chords.length === 0) {
      toast.error("No audio data to play");
      return;
    }

    setIsPlaying(true);
    
    try {
      // Set volume
      audioSynth.setMasterVolume(isMuted ? 0 : volume[0]);
      
      if (chords.length > 0) {
        await audioSynth.playChordProgression(chords, 1.5);
      } else if (notes.length > 0) {
        // Play notes sequentially
        for (let i = 0; i < notes.length; i++) {
          const note = notes[i];
          setTimeout(() => {
            audioSynth.playNote(
              note.pitch, 
              note.duration || 0.5, 
              note.velocity || 80,
              'piano'
            );
          }, i * 300);
        }
        
        // Wait for all notes to finish
        const totalDuration = notes.length * 300 + 1000;
        await new Promise(resolve => setTimeout(resolve, totalDuration));
      }
      
      toast.success("Playback completed");
    } catch (error) {
      console.error("Playback error:", error);
      toast.error("Playback failed");
    } finally {
      setIsPlaying(false);
    }
  };

  const handleStop = () => {
    audioSynth.stopAll();
    setIsPlaying(false);
    toast.info("Playback stopped");
  };

  const handleVolumeChange = (newVolume: number[]) => {
    setVolume(newVolume);
    if (!isMuted) {
      audioSynth.setMasterVolume(newVolume[0]);
    }
  };

  const toggleMute = () => {
    const newMuted = !isMuted;
    setIsMuted(newMuted);
    audioSynth.setMasterVolume(newMuted ? 0 : volume[0]);
  };

  return (
    <Card className={className}>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">{title}</h4>
            <div className="flex gap-2">
              {notes.length > 0 && (
                <Badge variant="outline" className="text-xs">
                  {notes.length} notes
                </Badge>
              )}
              {chords.length > 0 && (
                <Badge variant="outline" className="text-xs">
                  {chords.length} chords
                </Badge>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              size="sm"
              variant={isPlaying ? "secondary" : "default"}
              onClick={handlePlay}
              disabled={disabled || (notes.length === 0 && chords.length === 0)}
              className="flex items-center gap-2"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" />
                  Playing
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Play
                </>
              )}
            </Button>

            {isPlaying && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleStop}
                className="flex items-center gap-2"
              >
                <Stop className="w-4 h-4" />
                Stop
              </Button>
            )}

            <div className="flex items-center gap-2 flex-1">
              <Button
                size="sm"
                variant="ghost"
                onClick={toggleMute}
                className="p-2"
              >
                {isMuted ? (
                  <VolumeX className="w-4 h-4" />
                ) : (
                  <Volume2 className="w-4 h-4" />
                )}
              </Button>
              
              <div className="flex-1 max-w-24">
                <Slider
                  value={isMuted ? [0] : volume}
                  onValueChange={handleVolumeChange}
                  max={1}
                  min={0}
                  step={0.05}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {chords.length > 0 && (
            <div className="text-sm text-muted-foreground">
              Chords: {chords.join(" → ")}
            </div>
          )}
          
          {notes.length > 0 && (
            <div className="text-sm text-muted-foreground">
              Preview: {notes.slice(0, 3).map(n => {
                const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
                const octave = Math.floor(n.pitch / 12) - 1;
                const noteIndex = n.pitch % 12;
                return `${noteNames[noteIndex]}${octave}`;
              }).join(", ")}
              {notes.length > 3 && ` +${notes.length - 3} more`}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}