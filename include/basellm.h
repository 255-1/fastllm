#pragma once
#include "fastllm.h"

typedef void(*RuntimeResult) (int index, const char* content);//ʵʱ���ɵ����ݻص� index: 0��ʼ�ظ���-1���λظ�����

namespace fastllm {
	class basellm {
	public:
		basellm() {};
		~basellm() {};

		virtual void LoadFromFile(const std::string &fileName) = 0; // ���ļ���ȡ
		// ����
		virtual int Forward(
			const Data &inputIds,
			const Data &attentionMask,
			const Data &positionIds,
			std::vector <std::pair <Data, Data> > &pastKeyValues) = 0;

		virtual std::string Response(const std::string& input, RuntimeResult retCb) = 0; // ���ݸ��������ݻظ�

		virtual void SaveLowBitModel(const std::string &fileName, int bit) {}; // �洢������ģ��

		virtual void WarmUp() {}; // Ԥ��

		virtual void RotatePosition2D(Data &data, const Data &positionIds) {}; // ��άλ�ñ���

		virtual void CausalMask(Data &data, int start) {}; // ���mask

		int embed_dim = 4096;
		int num_attention_heads = 32;
		int head_dim = embed_dim / num_attention_heads;
		const int max_positions = 2048;
		const int rotary_dim = 64;
		const float scale_attn = sqrt(head_dim);

		int block_cnt = 28;

		std::vector <std::vector <float> > sin, cos;

		WeightMap weight; // Ȩ��

	};
}