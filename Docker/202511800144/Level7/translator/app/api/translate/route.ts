import { NextRequest, NextResponse } from 'next/server';

// 后端vLLM服务器地址，从环境变量中获取，默认为本地开发地址
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const MODEL_NAME = 'Tencent/Hunyuan-MT-7B';

export async function POST(request: NextRequest) {
  try {
    // 获取请求体数据
    const body = await request.json();
    const {
      text,
      sourceLang,
      targetLang,
      temperature = 0,
      maxTokens = 512,
      endpoint = 'chat' // 可选值: 'chat' 或 'completion'
    } = body;

    if (!text || !sourceLang || !targetLang) {
      return NextResponse.json(
        { error: '缺少必要参数: text, sourceLang, targetLang' },
        { status: 400 }
      );
    }

    // 构建请求参数
    const headers = {
      'Content-Type': 'application/json'
    };

    let apiUrl = '';
    let payload: object = {};

    if (endpoint === 'chat') {
      // 使用 /v1/chat/completions 终结点
      apiUrl = `${BACKEND_URL}/v1/chat/completions`;
      payload = {
        model: MODEL_NAME,
        messages: [
          {
            role: 'system',
            content: `你是一名专业的翻译官，请将用户提供的${sourceLang}文本准确流畅地翻译成${targetLang}。`
          },
          {
            role: 'user',
            content: `请将以下${sourceLang}内容翻译成${targetLang}：${text}`
          }
        ],
        temperature,
        max_tokens: maxTokens
      };
    } else {
      // 使用 /v1/completions 终结点
      apiUrl = `${BACKEND_URL}/v1/completions`;
      const prompt = `请将以下${sourceLang}文本翻译成${targetLang}：${text}`;
      payload = {
        model: MODEL_NAME,
        prompt,
        temperature,
        max_tokens: maxTokens
      };
    }

    // 发送请求到后端服务器
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`后端请求失败: ${response.status} ${response.statusText}`);
    }

    // 处理响应
    const result = await response.json();
    let translatedText = '';

    if (endpoint === 'chat') {
      translatedText = result?.choices?.[0]?.message?.content || '';
    } else {
      translatedText = result?.choices?.[0]?.text || '';
    }

    if (!translatedText) {
      throw new Error('翻译结果为空');
    }

    return NextResponse.json({
      success: true,
      translatedText
    });
  } catch (error) {
    console.error('翻译过程中出错:', error);
    return NextResponse.json(
      { 
        success: false,
        error: error instanceof Error ? error.message : '未知错误'
      },
      { status: 500 }
    );
  }
}