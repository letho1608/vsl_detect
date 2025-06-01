"""
Audio management for Vietnamese Sign Language Detection System.
"""

import os
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional

import gtts
from pygame import mixer

from .config import Config
from .logger import get_logger, log_performance


class AudioManager:
    """Manager for text-to-speech and audio playback."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Audio state
        self.is_speaking = False
        self.last_spoken_text = ""
        self.voice_dir = Path(config.get_absolute_path(config.audio.voice_dir))
        
        # Create voice directory
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame mixer
        try:
            mixer.init()
            self.logger.info("Audio manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def speak_text(self, text: str, force: bool = False) -> None:
        """
        Speak text using text-to-speech.
        
        Args:
            text: Text to speak
            force: Force speaking even if already speaking
        """
        if not text.strip():
            return
        
        if self.is_speaking and not force:
            self.logger.debug("Already speaking, skipping")
            return
        
        if text == self.last_spoken_text and not force:
            self.logger.debug("Same text as last spoken, skipping")
            return
        
        # Start speaking in background thread
        thread = threading.Thread(
            target=self._speak_thread,
            args=(text,),
            daemon=True
        )
        thread.start()
    
    @log_performance("text_to_speech")
    def _speak_thread(self, text: str) -> None:
        """
        Background thread for text-to-speech processing.
        
        Args:
            text: Text to speak
        """
        audio_path = None
        try:
            self.is_speaking = True
            self.last_spoken_text = text
            
            self.logger.debug(f"Speaking: {text}")
            
            # Generate unique filename
            timestamp = str(int(time.time() * 1000))
            safe_filename = f'speech_{timestamp}.mp3'
            audio_path = self.voice_dir / safe_filename
            
            # Generate TTS audio
            tts = gtts.gTTS(text=text, lang=self.config.audio.language)
            tts.save(str(audio_path))
            
            # Verify file was created
            if not audio_path.exists():
                raise FileNotFoundError("Audio file was not created")
            
            # Small delay to ensure file is written
            time.sleep(0.1)
            
            # Play audio
            self._play_audio(str(audio_path))
            
        except Exception as e:
            self.logger.error(f"Text-to-speech error: {e}")
        finally:
            self.is_speaking = False
            # Cleanup audio file
            if audio_path and audio_path.exists() and self.config.audio.temp_audio_cleanup:
                try:
                    audio_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup audio file: {e}")
    
    def _play_audio(self, audio_path: str) -> None:
        """
        Play audio file using pygame mixer.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            mixer.music.load(audio_path)
            mixer.music.play()
            
            # Wait for playback to complete
            while mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
        finally:
            try:
                mixer.music.unload()
            except:
                pass  # Ignore unload errors
    
    def stop_speaking(self) -> None:
        """Stop current speech playback."""
        try:
            mixer.music.stop()
            self.is_speaking = False
            self.logger.debug("Speech stopped")
        except Exception as e:
            self.logger.warning(f"Failed to stop speech: {e}")
    
    def is_audio_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.is_speaking or mixer.music.get_busy()
    
    def cleanup(self) -> None:
        """Cleanup audio resources."""
        try:
            self.stop_speaking()
            mixer.quit()
            
            # Cleanup temporary audio files
            if self.config.audio.temp_audio_cleanup:
                for audio_file in self.voice_dir.glob("speech_*.mp3"):
                    try:
                        audio_file.unlink()
                    except:
                        pass
            
            self.logger.info("Audio manager cleaned up")
        except Exception as e:
            self.logger.warning(f"Audio cleanup error: {e}")


class AutoSpeaker:
    """Automatic speech controller for accumulated text."""
    
    def __init__(self, audio_manager: AudioManager, config: Config):
        self.audio_manager = audio_manager
        self.config = config
        self.logger = get_logger(__name__)
        
        # Auto-speak state
        self.current_sentence = []
        self.is_auto_speak_enabled = config.audio.auto_speak
        self.auto_speak_threshold = config.audio.auto_speak_threshold
    
    def add_word(self, word: str) -> None:
        """
        Add word to current sentence and speak when threshold reached.
        
        Args:
            word: Word to add
        """
        if not word.strip():
            return
        
        self.current_sentence.append(word.strip())
        
        # Auto-speak if enabled and threshold reached
        if (self.is_auto_speak_enabled and 
            len(self.current_sentence) >= self.auto_speak_threshold):
            self.speak_sentence()
    
    def speak_sentence(self) -> None:
        """Speak current sentence and clear it."""
        if not self.current_sentence:
            return
        
        sentence = " ".join(self.current_sentence)
        self.audio_manager.speak_text(sentence)
        self.clear_sentence()
    
    def clear_sentence(self) -> None:
        """Clear current sentence."""
        self.current_sentence.clear()
        self.logger.debug("Sentence cleared")
    
    def get_current_text(self) -> str:
        """Get current accumulated text."""
        return " ".join(self.current_sentence)
    
    def set_auto_speak(self, enabled: bool) -> None:
        """Enable/disable auto-speak."""
        self.is_auto_speak_enabled = enabled
        self.logger.info(f"Auto-speak {'enabled' if enabled else 'disabled'}")
    
    def set_threshold(self, threshold: int) -> None:
        """Set auto-speak word threshold."""
        self.auto_speak_threshold = max(1, threshold)
        self.logger.info(f"Auto-speak threshold set to {self.auto_speak_threshold}")


# Convenience functions
def create_audio_manager(config: Config) -> AudioManager:
    """Create and return audio manager instance."""
    return AudioManager(config)


def create_auto_speaker(audio_manager: AudioManager, config: Config) -> AutoSpeaker:
    """Create and return auto speaker instance."""
    return AutoSpeaker(audio_manager, config)