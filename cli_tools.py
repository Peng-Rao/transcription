import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from transcription.audio_extractor import AudioExtractor
from transcription.latex_generator import LaTeXGenerator
from transcription.text_processor import TextProcessor
from transcription.transcriber import Transcriber

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_audio_cli():
    """命令行工具：从视频中提取音频"""
    parser = argparse.ArgumentParser(
        description="从视频文件中提取音频",
        prog="extract-audio"
    )
    parser.add_argument("video_path", help="输入视频文件路径")
    parser.add_argument("-o", "--output", help="输出音频文件路径 (默认: video_name_audio.wav)")
    parser.add_argument("--sample-rate", type=int, default=16000, 
                       help="音频采样率 (默认: 16000)")
    
    args = parser.parse_args()
    
    try:
        extractor = AudioExtractor()
        
        # 确定输出路径
        if args.output:
            audio_path = args.output
        else:
            video_path = Path(args.video_path)
            audio_path = f"{video_path.stem}_audio.wav"
        
        extractor.extract_audio(args.video_path, audio_path, args.sample_rate)
        print(f"音频提取成功: {audio_path}")
        
    except Exception as e:
        logger.error(f"音频提取失败: {e}")
        sys.exit(1)


def transcribe_audio_cli():
    """命令行工具：将音频转录为文本"""
    parser = argparse.ArgumentParser(
        description="使用Whisper将音频转录为文本",
        prog="transcribe-audio"
    )
    parser.add_argument("audio_path", help="输入音频文件路径")
    parser.add_argument("-o", "--output", help="输出SRT字幕文件路径 (默认: audio_name.srt)")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("--language", help="语言代码 (如: en, zh, es)")
    parser.add_argument("--text-only", action="store_true", 
                       help="只输出纯文本到控制台")
    parser.add_argument("--info", action="store_true", 
                       help="显示转录信息")
    
    args = parser.parse_args()
    
    try:
        transcriber = Transcriber(model_size=args.model)
        
        if args.info:
            # 显示转录信息
            info = transcriber.get_transcription_info(args.audio_path)
            print("\n=== 转录信息 ===")
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        # 确定输出路径
        if args.output:
            srt_path = args.output
        else:
            audio_path = Path(args.audio_path)
            srt_path = f"{audio_path.stem}.srt"
        
        # 转录音频
        transcriber.transcribe_to_srt(args.audio_path, srt_path, args.language)
        
        if args.text_only:
            # 提取并显示纯文本
            text = transcriber.extract_text_from_srt(srt_path)
            print("\n=== 转录文本 ===")
            print(text)
        else:
            print(f"转录完成: {srt_path}")
            
    except Exception as e:
        logger.error(f"转录失败: {e}")
        sys.exit(1)


def process_text_cli():
    """命令行工具：处理和清理文本"""
    parser = argparse.ArgumentParser(
        description="处理和清理转录文本",
        prog="process-text"
    )
    parser.add_argument("input_path", help="输入文本文件路径 (支持.txt或.srt)")
    parser.add_argument("-o", "--output", help="输出处理后的文本文件路径")
    parser.add_argument("--from-srt", action="store_true", 
                       help="从SRT文件中提取文本")
    parser.add_argument("--preview", action="store_true", 
                       help="预览处理后的文本")
    
    args = parser.parse_args()
    
    try:
        processor = TextProcessor()
        
        # 读取输入文本
        if args.from_srt or args.input_path.endswith('.srt'):
            # 从SRT文件提取文本
            transcriber = Transcriber()
            raw_text = transcriber.extract_text_from_srt(args.input_path)
        else:
            # 从普通文本文件读取
            with open(args.input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        
        # 处理文本
        processed_text = processor.process_text(raw_text)
        
        if args.preview:
            print("\n=== 处理后的文本预览 ===")
            print(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
            print(f"\n文本长度: {len(processed_text)} 字符")
            return
        
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_path)
            output_path = f"{input_path.stem}_processed.txt"
        
        # 保存处理后的文本
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        print(f"文本处理完成: {output_path}")
        
    except Exception as e:
        logger.error(f"文本处理失败: {e}")
        sys.exit(1)


def generate_latex_cli():
    """命令行工具：生成LaTeX笔记"""
    parser = argparse.ArgumentParser(
        description="从处理后的文本生成LaTeX笔记",
        prog="generate-latex"
    )
    parser.add_argument("text_path", help="输入文本文件路径")
    parser.add_argument("-o", "--output", help="输出LaTeX文件路径")
    parser.add_argument("--title", default="Lecture Notes", help="笔记标题")
    parser.add_argument("--model", default="deepseek-reasoner", 
                       help="使用的AI模型")
    parser.add_argument("--template-only", action="store_true", 
                       help="只使用模板生成，不调用AI")
    parser.add_argument("--api-key", help="API密钥 (或从环境变量DEEPSEEK_API_KEY读取)")
    
    args = parser.parse_args()
    
    try:
        # 加载环境变量
        load_dotenv()
        
        # 获取API密钥
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key and not args.template_only:
            logger.warning("未提供API密钥，将使用模板生成")
            api_key = None
        
        # 读取文本
        with open(args.text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 确定输出路径
        if args.output:
            latex_path = args.output
        else:
            text_path = Path(args.text_path)
            latex_path = f"{text_path.stem}_notes.tex"
        
        # 生成LaTeX
        generator = LaTeXGenerator(api_key=api_key, model=args.model)
        generator.generate_notes(text, latex_path, args.title)
        
        print(f"LaTeX笔记生成完成: {latex_path}")
        
        # 提供编译建议
        print("\n编译建议:")
        print(f"pdflatex {latex_path}")
        print(f"或者: xelatex {latex_path}")
        
    except Exception as e:
        logger.error(f"LaTeX生成失败: {e}")
        sys.exit(1)


def batch_process_cli():
    """命令行工具：批量处理多个文件"""
    parser = argparse.ArgumentParser(
        description="批量处理多个视频文件",
        prog="batch-process"
    )
    parser.add_argument("input_dir", help="输入目录路径")
    parser.add_argument("-o", "--output", default="batch_output", 
                       help="输出目录路径")
    parser.add_argument("--extensions", default="mp4,avi,mkv,mov", 
                       help="支持的视频文件扩展名 (逗号分隔)")
    parser.add_argument("--whisper-model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", help="语言代码")
    parser.add_argument("--keep-intermediates", action="store_true", 
                       help="保留中间文件")
    
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # 获取支持的扩展名
        extensions = [ext.strip().lower() for ext in args.extensions.split(',')]
        
        # 查找视频文件
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f"*.{ext}"))
        
        if not video_files:
            print(f"在 {input_dir} 中未找到支持的视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        # 初始化处理器
        from main import LectureProcessor
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        processor = LectureProcessor(api_key=api_key, whisper_model=args.whisper_model)
        
        # 批量处理
        success_count = 0
        for video_file in video_files:
            try:
                print(f"\n处理: {video_file.name}")
                latex_file = processor.process_lecture(
                    str(video_file),
                    str(output_dir),
                    args.keep_intermediates,
                    args.language
                )
                print(f"完成: {latex_file}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理 {video_file.name} 失败: {e}")
                continue
        
        print(f"\n批量处理完成: {success_count}/{len(video_files)} 个文件成功")
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        sys.exit(1)


def main():
    """主命令行入口，根据程序名选择功能"""
    prog_name = Path(sys.argv[0]).stem
    
    # 根据程序名或参数选择功能
    if prog_name == "extract-audio" or (len(sys.argv) > 1 and sys.argv[1] == "extract-audio"):
        if len(sys.argv) > 1 and sys.argv[1] == "extract-audio":
            sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除子命令
        extract_audio_cli()
    elif prog_name == "transcribe-audio" or (len(sys.argv) > 1 and sys.argv[1] == "transcribe-audio"):
        if len(sys.argv) > 1 and sys.argv[1] == "transcribe-audio":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        transcribe_audio_cli()
    elif prog_name == "process-text" or (len(sys.argv) > 1 and sys.argv[1] == "process-text"):
        if len(sys.argv) > 1 and sys.argv[1] == "process-text":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        process_text_cli()
    elif prog_name == "generate-latex" or (len(sys.argv) > 1 and sys.argv[1] == "generate-latex"):
        if len(sys.argv) > 1 and sys.argv[1] == "generate-latex":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        generate_latex_cli()
    elif prog_name == "batch-process" or (len(sys.argv) > 1 and sys.argv[1] == "batch-process"):
        if len(sys.argv) > 1 and sys.argv[1] == "batch-process":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        batch_process_cli()
    else:
        # 显示帮助信息
        print("Lecture Processing Tools - 独立命令行工具")
        print("\n可用命令:")
        print("  extract-audio    - 从视频中提取音频")
        print("  transcribe-audio - 将音频转录为文本")
        print("  process-text     - 处理和清理文本")
        print("  generate-latex   - 生成LaTeX笔记")
        print("  batch-process    - 批量处理多个文件")
        print("\n使用方法:")
        print("  python cli_tools.py <命令> [参数]")
        print("  或者直接: python -m cli_tools <命令> [参数]")
        print("\n获取具体命令帮助:")
        print("  python cli_tools.py <命令> --help")


if __name__ == "__main__":
    main()