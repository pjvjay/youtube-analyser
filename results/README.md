# Where someone is standing: results for all downloaded videos

The standing detector (`standing/detect_standing.py`) analysed **248 videos (210.4 hours)** at 1 frame per second. It reported **120 standing sections**, after dropping 23 sections that were statues, posters or artwork (figures that never moved). Every one of the remaining sections was checked by eye:

| Verdict | Sections | Meaning |
|---|---|---|
| confirmed | 105 | Someone is standing |
| unclear | 6 | A figure that cannot be made out at 144p |
| false alarm | 9 | A guest sitting square-on to the camera |

Confirmed standing appears in **47 videos** and adds up to **11.5 minutes**. What it shows:

| What | Sections |
|---|---|
| Bystander standing at the expo booth | 46 |
| Person standing in the studio | 29 |
| Opening sequence | 26 |
| Person walking in the garden behind the guest | 4 |

The full list, with titles, YouTube links that open at each section and the verdict, is in
[`standing_sections.csv`](standing_sections.csv). The table below has the same sections
except the false alarms.

## How well it works

- **Precision**: 105 of 120 reported sections are confirmed (88%), 92% if the unclear ones are real. Before the still-figure filter it was about 73%.
- **Misses**: 112 frames the detector did not flag were checked by eye:
  - 48 frames where someone's legs were visible: 2 missed standing people (one only half in frame, one 2 seconds before an opening the detector did catch).
  - 32 frames from the first 20 seconds of episodes: 3 to 5 missed, all waist-up shots of standing people whose legs are out of frame.
  - 32 random frames: no clear misses, one ambiguous (people at an expo table).
- **So**: when a person's legs are in frame, standing is found reliably. Waist-up shots of standing people are not, by design, because without legs the rules cannot tell standing from sitting facing the camera.

Known false-alarm pattern: a guest sitting alone, square-on to the camera, can look straight-legged in 2D (9 sections, in 2 episodes). With a seated person next to them for comparison this is already caught.

## Sections by video (confirmed and unclear)

| Uploaded | Video | Title | Start | End | Seconds | Verdict | Note |
|---|---|---|---|---|---|---|---|
| 2019-11-15 | `20191115_ohIo2d7eKV4.mp4` | Karl Kani / S01 EP1 / Full Episode | [0:00:12](https://www.youtube.com/watch?v=ohIo2d7eKV4&t=12s) | 0:00:14 | 2.0 | confirmed | opening sequence |
| 2019-11-22 | `20191122_R8BHKLb0gek.mp4` | Rosa Acosta / S01 EP2 / Full Episode | [0:00:11](https://www.youtube.com/watch?v=R8BHKLb0gek&t=11s) | 0:00:15 | 4.0 | confirmed | opening sequence |
| 2020-01-10 | `20200110_KuAFwHfkdqo.mp4` | Rude Jude / S01 EP9 / (Part 1) | [0:00:12](https://www.youtube.com/watch?v=KuAFwHfkdqo&t=12s) | 0:00:14 | 2.0 | confirmed | opening sequence |
| 2020-01-17 | `20200117_M49Fr57W_Uo.mp4` | Rude Jude / S01 EP9 (PT.2) / Full Episode | [0:00:12](https://www.youtube.com/watch?v=M49Fr57W_Uo&t=12s) | 0:00:14 | 2.0 | confirmed | opening sequence |
| 2020-02-20 | `20200220_vJOCSwJzbOo.mp4` | NeYo / S01 EP13 / Part 2 | [0:00:02](https://www.youtube.com/watch?v=vJOCSwJzbOo&t=2s) | 0:00:04 | 2.0 | confirmed | opening sequence |
| 2020-03-20 | `20200320_df7uyaUUtdU.mp4` | James "The Beast" Wilson / S01 EP17 / | [0:00:02](https://www.youtube.com/watch?v=df7uyaUUtdU&t=2s) | 0:00:04 | 2.0 | confirmed | opening sequence |
| 2020-05-11 | `20200511_BBN61h_8Mu8.mp4` | Private Talk / Best Of Pt.2 | [0:00:08](https://www.youtube.com/watch?v=BBN61h_8Mu8&t=8s) | 0:00:11 | 3.0 | confirmed | opening sequence |
| 2020-06-06 | `20200606_hahGcCTPJiM.mp4` | JADA STEVENS | [0:00:16](https://www.youtube.com/watch?v=hahGcCTPJiM&t=16s) | 0:00:18 | 2.0 | confirmed | opening sequence |
| 2020-06-06 | `20200606_hahGcCTPJiM.mp4` | JADA STEVENS | [0:15:50](https://www.youtube.com/watch?v=hahGcCTPJiM&t=950s) | 0:15:56 | 6.0 | confirmed |  |
| 2020-12-16 | `20201216_t7r4l7bM36A.mp4` | DANII BANKS | [0:56:07](https://www.youtube.com/watch?v=t7r4l7bM36A&t=3367s) | 0:56:17 | 10.0 | confirmed |  |
| 2020-12-16 | `20201216_t7r4l7bM36A.mp4` | DANII BANKS | [0:56:31](https://www.youtube.com/watch?v=t7r4l7bM36A&t=3391s) | 0:56:57 | 26.0 | confirmed |  |
| 2021-01-27 | `20210127_Ie3I3zyIV4Y.mp4` | PRIVATE TALK AFTER DARK / EP7 / LEXY PANTERRA | [0:33:59](https://www.youtube.com/watch?v=Ie3I3zyIV4Y&t=2039s) | 0:34:05 | 6.0 | unclear | dark figure in the unlit background behind the host; it moves a little but cannot be made out at 144p |
| 2021-01-27 | `20210127_Ie3I3zyIV4Y.mp4` | PRIVATE TALK AFTER DARK / EP7 / LEXY PANTERRA | [0:37:29](https://www.youtube.com/watch?v=Ie3I3zyIV4Y&t=2249s) | 0:37:31 | 2.0 | confirmed |  |
| 2021-01-27 | `20210127_Ie3I3zyIV4Y.mp4` | PRIVATE TALK AFTER DARK / EP7 / LEXY PANTERRA | [0:37:34](https://www.youtube.com/watch?v=Ie3I3zyIV4Y&t=2254s) | 0:37:37 | 3.0 | confirmed |  |
| 2021-05-06 | `20210506_x99sNANMy8o.mp4` | ANA FOXXX / EP 39 (After Dark) | [0:39:19](https://www.youtube.com/watch?v=x99sNANMy8o&t=2359s) | 0:39:21 | 2.0 | confirmed |  |
| 2021-05-27 | `20210527_76-ybvHY5-Y.mp4` | MC EIHT / EP.42 | [0:05:21](https://www.youtube.com/watch?v=76-ybvHY5-Y&t=321s) | 0:05:26 | 5.0 | confirmed |  |
| 2021-07-01 | `20210701_hZvWEen3O7Y.mp4` | CANDACE CANE / EP 47 | [0:44:21](https://www.youtube.com/watch?v=hZvWEen3O7Y&t=2661s) | 0:44:27 | 6.0 | confirmed |  |
| 2021-07-29 | `20210729_3DCvTY2A7sY.mp4` | BIG NEECHI / EP 51 | [0:32:44](https://www.youtube.com/watch?v=3DCvTY2A7sY&t=1964s) | 0:32:46 | 2.0 | confirmed |  |
| 2021-09-09 | `20210909_8w6Gcsknt3M.mp4` | MARIE LUV (After Dark) / EP 57 | [0:13:55](https://www.youtube.com/watch?v=8w6Gcsknt3M&t=835s) | 0:14:07 | 12.0 | confirmed |  |
| 2021-09-09 | `20210909_8w6Gcsknt3M.mp4` | MARIE LUV (After Dark) / EP 57 | [0:54:52](https://www.youtube.com/watch?v=8w6Gcsknt3M&t=3292s) | 0:54:54 | 2.0 | confirmed |  |
| 2021-11-25 | `20211125_hbeb1YRh4Qw.mp4` | Samantha Saint / EP 65 (After Dark) | [0:57:40](https://www.youtube.com/watch?v=hbeb1YRh4Qw&t=3460s) | 0:57:42 | 2.0 | confirmed |  |
| 2021-11-25 | `20211125_hbeb1YRh4Qw.mp4` | Samantha Saint / EP 65 (After Dark) | [0:57:58](https://www.youtube.com/watch?v=hbeb1YRh4Qw&t=3478s) | 0:58:06 | 8.0 | confirmed |  |
| 2021-12-02 | `20211202_Lklec-suWsQ.mp4` | DAISY MARIE / EP 66 (After Dark) | [0:42:00](https://www.youtube.com/watch?v=Lklec-suWsQ&t=2520s) | 0:42:04 | 4.0 | confirmed |  |
| 2021-12-02 | `20211202_Lklec-suWsQ.mp4` | DAISY MARIE / EP 66 (After Dark) | [0:42:08](https://www.youtube.com/watch?v=Lklec-suWsQ&t=2528s) | 0:42:14 | 6.0 | confirmed |  |
| 2021-12-02 | `20211202_Lklec-suWsQ.mp4` | DAISY MARIE / EP 66 (After Dark) | [0:42:17](https://www.youtube.com/watch?v=Lklec-suWsQ&t=2537s) | 0:42:21 | 4.0 | confirmed |  |
| 2021-12-09 | `20211209_kWih-qVfWio.mp4` | JUHAHN JONES / EP 67 | [0:04:18](https://www.youtube.com/watch?v=kWih-qVfWio&t=258s) | 0:04:21 | 3.0 | confirmed |  |
| 2022-04-14 | `20220414_4jnKepvUk60.mp4` | NIKKI DELANO / EP 85 (After Dark) | [0:23:51](https://www.youtube.com/watch?v=4jnKepvUk60&t=1431s) | 0:23:58 | 7.0 | unclear | figure-like shape behind the host; cannot be made out at 144p |
| 2022-05-26 | `20220526__rMkKOH2BU4.mp4` | GOOD MOMS BAD CHOICES / EP 90 | [0:53:57](https://www.youtube.com/watch?v=_rMkKOH2BU4&t=3237s) | 0:53:59 | 2.0 | unclear | guest perched on the edge of the couch or standing in front of it |
| 2022-06-09 | `20220609_2GEgiylsnPg.mp4` | KAZUMI / EP 92 (After Dark) | [0:03:31](https://www.youtube.com/watch?v=2GEgiylsnPg&t=211s) | 0:03:33 | 2.0 | confirmed |  |
| 2022-08-08 | `20220808_ILFEJDqJhnE.mp4` | JESSY JONES / EP. 95 (AFTER DARK) | [1:18:02](https://www.youtube.com/watch?v=ILFEJDqJhnE&t=4682s) | 1:18:05 | 3.0 | confirmed |  |
| 2022-08-08 | `20220808_ILFEJDqJhnE.mp4` | JESSY JONES / EP. 95 (AFTER DARK) | [1:20:46](https://www.youtube.com/watch?v=ILFEJDqJhnE&t=4846s) | 1:20:53 | 7.0 | confirmed |  |
| 2022-08-24 | `20220824_mUur6IWfuVY.mp4` | KING HEFF (After Dark - Exxotica Edition) / EP 101 | [0:00:03](https://www.youtube.com/watch?v=mUur6IWfuVY&t=3s) | 0:00:05 | 2.0 | confirmed | opening sequence |
| 2022-08-26 | `20220826_urIm_CYdasg.mp4` | CAITLIN BELL & KARMEN KARMA (After Dark) / EP 98 / FULL EPISODE / PATREON EXCLUSIVE | [1:20:16](https://www.youtube.com/watch?v=urIm_CYdasg&t=4816s) | 1:20:19 | 3.0 | confirmed |  |
| 2022-08-26 | `20220826_urIm_CYdasg.mp4` | CAITLIN BELL & KARMEN KARMA (After Dark) / EP 98 / FULL EPISODE / PATREON EXCLUSIVE | [1:20:31](https://www.youtube.com/watch?v=urIm_CYdasg&t=4831s) | 1:20:35 | 4.0 | confirmed |  |
| 2022-08-26 | `20220826_urIm_CYdasg.mp4` | CAITLIN BELL & KARMEN KARMA (After Dark) / EP 98 / FULL EPISODE / PATREON EXCLUSIVE | [1:35:59](https://www.youtube.com/watch?v=urIm_CYdasg&t=5759s) | 1:36:03 | 4.0 | confirmed |  |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:01:20](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=80s) | 0:01:31 | 11.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:01:52](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=112s) | 0:01:57 | 5.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:13:50](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=830s) | 0:13:52 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:14:11](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=851s) | 0:14:14 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:14:37](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=877s) | 0:14:58 | 21.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:15:06](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=906s) | 0:15:09 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:15:15](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=915s) | 0:15:22 | 7.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:15:58](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=958s) | 0:16:36 | 38.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:16:39](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=999s) | 0:16:59 | 20.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:17:02](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1022s) | 0:17:04 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:17:08](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1028s) | 0:17:16 | 8.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:17:21](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1041s) | 0:17:24 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:17:44](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1064s) | 0:17:51 | 7.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:18:02](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1082s) | 0:18:12 | 10.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:18:15](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1095s) | 0:18:24 | 9.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:18:32](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1112s) | 0:18:35 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:18:44](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1124s) | 0:19:37 | 53.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:20:25](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1225s) | 0:20:30 | 5.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:20:33](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1233s) | 0:20:41 | 8.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:20:47](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1247s) | 0:20:52 | 5.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:20:56](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1256s) | 0:20:58 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:21:18](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1278s) | 0:22:47 | 89.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:22:55](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1375s) | 0:23:04 | 9.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:24:58](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1498s) | 0:25:06 | 8.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:25:11](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1511s) | 0:25:20 | 9.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:25:59](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1559s) | 0:26:03 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:26:37](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1597s) | 0:26:40 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:27:09](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1629s) | 0:27:13 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:27:18](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1638s) | 0:27:22 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:28:23](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1703s) | 0:28:33 | 10.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:28:58](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1738s) | 0:29:00 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:30:29](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1829s) | 0:30:33 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:31:02](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1862s) | 0:31:05 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:31:17](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1877s) | 0:31:23 | 6.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:31:50](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1910s) | 0:31:52 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:32:17](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1937s) | 0:32:24 | 7.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:32:32](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1952s) | 0:32:34 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:32:44](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1964s) | 0:32:51 | 7.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:33:02](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1982s) | 0:33:05 | 3.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:33:15](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=1995s) | 0:33:19 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:33:26](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=2006s) | 0:33:32 | 6.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:33:39](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=2019s) | 0:33:43 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:33:55](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=2035s) | 0:34:00 | 5.0 | confirmed | bystander standing at the expo booth |
| 2022-09-08 | `20220908_Yfz0_xX9-yU.mp4` | SAFAREE / S03 EP. 103 / FULL EPISODE | [0:34:08](https://www.youtube.com/watch?v=Yfz0_xX9-yU&t=2048s) | 0:34:14 | 6.0 | confirmed | bystander standing at the expo booth |
| 2022-09-22 | `20220922_Pu_pzRPqf08.mp4` | ZACK WOODS / EP. 105 (EXXXOTICA EDITION) | [0:16:28](https://www.youtube.com/watch?v=Pu_pzRPqf08&t=988s) | 0:16:32 | 4.0 | confirmed | bystander standing at the expo booth |
| 2022-09-22 | `20220922_Pu_pzRPqf08.mp4` | ZACK WOODS / EP. 105 (EXXXOTICA EDITION) | [0:30:20](https://www.youtube.com/watch?v=Pu_pzRPqf08&t=1820s) | 0:30:22 | 2.0 | confirmed | bystander standing at the expo booth |
| 2022-10-19 | `20221019_neqw9WZLduo.mp4` | ALEXAS MORGAN (After Dark) / EP 104 (EXXXOTICA EDITION) | [0:00:08](https://www.youtube.com/watch?v=neqw9WZLduo&t=8s) | 0:00:10 | 2.0 | confirmed | opening sequence |
| 2023-08-30 | `20230830_AV60ApVf23M.mp4` | RARA KNUPPS / EP. 139 | [0:57:31](https://www.youtube.com/watch?v=AV60ApVf23M&t=3451s) | 0:57:34 | 3.0 | confirmed |  |
| 2023-10-19 | `20231019_ywRF7GgJvGY.mp4` | BEST OF S04 / BONUS EPISODE (Pt.1) | [0:30:23](https://www.youtube.com/watch?v=ywRF7GgJvGY&t=1823s) | 0:30:26 | 3.0 | confirmed |  |
| 2023-11-29 | `20231129_Yhg3ZnALVAE.mp4` | RE-RUN FLASHBACK EPISODE / KAZUMI | [0:03:56](https://www.youtube.com/watch?v=Yhg3ZnALVAE&t=236s) | 0:04:00 | 4.0 | confirmed |  |
| 2023-12-14 | `20231214_MZwwyHibf_M.mp4` | RE-RUN FLASHBACK EPISODE / ADAM 22 & LENA THE PLUG | [0:06:00](https://www.youtube.com/watch?v=MZwwyHibf_M&t=360s) | 0:06:05 | 5.0 | unclear | large dark figure behind the host; may be a person or a wall hanging |
| 2023-12-14 | `20231214_MZwwyHibf_M.mp4` | RE-RUN FLASHBACK EPISODE / ADAM 22 & LENA THE PLUG | [0:27:42](https://www.youtube.com/watch?v=MZwwyHibf_M&t=1662s) | 0:27:47 | 5.0 | unclear | same figure as at 6:00 |
| 2024-01-06 | `20240106_meUa9mraazo.mp4` | RE-RUN FLASHBACK EPISODE / JADA STEVENS | [0:16:16](https://www.youtube.com/watch?v=meUa9mraazo&t=976s) | 0:16:22 | 6.0 | confirmed |  |
| 2024-09-30 | `20240930_NruWvgHm2z8.mp4` | RE-RUN FLASHBACK EPISODE / CJ SPARXX S03 EP88 | [0:54:18](https://www.youtube.com/watch?v=NruWvgHm2z8&t=3258s) | 0:54:20 | 2.0 | unclear | partial dark shape at the left edge |
| 2025-01-07 | `20250107_gkv8JExVZRs.mp4` | SARA JAY / S06 - EP 184 | [0:00:14](https://www.youtube.com/watch?v=gkv8JExVZRs&t=14s) | 0:00:19 | 5.0 | confirmed | opening sequence |
| 2025-01-22 | `20250122_AHTN50ck5dk.mp4` | KEN STARZ / EP 185 | [0:00:14](https://www.youtube.com/watch?v=AHTN50ck5dk&t=14s) | 0:00:19 | 5.0 | confirmed | opening sequence |
| 2025-02-21 | `20250221_Au3rpg7iLTY.webm` | HONEY BXBY / EP 186 | [0:00:15](https://www.youtube.com/watch?v=Au3rpg7iLTY&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-03-01 | `20250301_qdcOYZs4Ue0.mkv` | MR PAPERS / S06 EP 186 | [0:00:15](https://www.youtube.com/watch?v=qdcOYZs4Ue0&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-03-08 | `20250308_70g1532F4Ec.mp4` | WOLFACE JOEY / EP 188 | [0:00:15](https://www.youtube.com/watch?v=70g1532F4Ec&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-03-08 | `20250308_70g1532F4Ec.mp4` | WOLFACE JOEY / EP 188 | [0:20:42](https://www.youtube.com/watch?v=70g1532F4Ec&t=1242s) | 0:20:46 | 4.0 | confirmed |  |
| 2025-03-22 | `20250322_fS-JtFwYAAQ.mp4` | MAZEE THE GOAT / EP 189 | [0:00:15](https://www.youtube.com/watch?v=fS-JtFwYAAQ&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-03-31 | `20250331_63oM4Rr-mS0.mkv` | NU JERZEY DEVIL / EP 190 | [0:00:15](https://www.youtube.com/watch?v=63oM4Rr-mS0&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-03-31 | `20250331_63oM4Rr-mS0.mkv` | NU JERZEY DEVIL / EP 190 | [1:27:43](https://www.youtube.com/watch?v=63oM4Rr-mS0&t=5263s) | 1:27:47 | 4.0 | confirmed |  |
| 2025-04-17 | `20250417_fObml4HrMv8.mkv` | SAM JAY / EP 191 | [0:00:15](https://www.youtube.com/watch?v=fObml4HrMv8&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-05-16 | `20250516_Qc3ynwRH_Io.webm` | JOANNA ANGEL / EP 192 | [0:00:15](https://www.youtube.com/watch?v=Qc3ynwRH_Io&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-05-22 | `20250522_hj1vtoznmGA.mp4` | FARRAH ABRAHAM / EP 193 | [0:00:15](https://www.youtube.com/watch?v=hj1vtoznmGA&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-06-02 | `20250602_ZvXFXEre76w.mkv` | BRANDI GLANVILLE | [0:00:15](https://www.youtube.com/watch?v=ZvXFXEre76w&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-06-14 | `20250614_e654H4tIxWY.mkv` | DR JOSHUA GONZALEZ / EP 195 | [0:00:13](https://www.youtube.com/watch?v=e654H4tIxWY&t=13s) | 0:00:19 | 6.0 | confirmed | opening sequence |
| 2025-06-21 | `20250621_wkrStVa4u1o.mkv` | AMOURANTH / EP 196 | [0:00:15](https://www.youtube.com/watch?v=wkrStVa4u1o&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-08-08 | `20250808_eQVB4ifCz-o.mp4` | B JACKS / EP.197 | [0:00:14](https://www.youtube.com/watch?v=eQVB4ifCz-o&t=14s) | 0:00:19 | 5.0 | confirmed | opening sequence |
| 2025-08-29 | `20250829_x_-iMlUvua4.mp4` | THE FLESH MECHANIC / EP 198 | [0:00:15](https://www.youtube.com/watch?v=x_-iMlUvua4&t=15s) | 0:00:19 | 4.0 | confirmed | opening sequence |
| 2025-09-05 | `20250905_8DzbNEp-heo.mkv` | MESA aka MR. 100 SUMMERS | [0:00:15](https://www.youtube.com/watch?v=8DzbNEp-heo&t=15s) | 0:00:20 | 5.0 | confirmed | opening sequence |
| 2026-06-19 | `20260619_l_1-R4oJ-pI.webm` | Silvia Saige & Sarita Natividad / EP 206 | [0:00:17](https://www.youtube.com/watch?v=l_1-R4oJ-pI&t=17s) | 0:00:19 | 2.0 | confirmed | person walking in the garden behind the guest |
| 2026-06-19 | `20260619_l_1-R4oJ-pI.webm` | Silvia Saige & Sarita Natividad / EP 206 | [0:14:26](https://www.youtube.com/watch?v=l_1-R4oJ-pI&t=866s) | 0:14:29 | 3.0 | confirmed | person walking in the garden behind the guest |
| 2026-06-19 | `20260619_l_1-R4oJ-pI.webm` | Silvia Saige & Sarita Natividad / EP 206 | [0:24:03](https://www.youtube.com/watch?v=l_1-R4oJ-pI&t=1443s) | 0:24:10 | 7.0 | confirmed | person walking in the garden behind the guest |
| 2026-06-26 | `20260626_UJi-bAAW70g.mp4` | Rachel Jade & Lauren Hunt / EP 207 | [0:30:46](https://www.youtube.com/watch?v=UJi-bAAW70g&t=1846s) | 0:30:48 | 2.0 | confirmed | person walking in the garden behind the guest |

## False alarms

| Video | Start | End | Note |
|---|---|---|---|
| `20220526__rMkKOH2BU4.mp4` | 1:03:20 | 1:03:23 | guest seated on the couch facing the camera with crossed legs |
| `20220526__rMkKOH2BU4.mp4` | 1:03:26 | 1:03:31 | guest seated on the couch facing the camera with crossed legs |
| `20220616_Dv_-QKO2E1M.mp4` | 0:31:16 | 0:31:18 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 0:34:13 | 0:34:15 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 0:41:14 | 0:41:17 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 0:41:26 | 0:41:29 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 1:02:41 | 1:02:44 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 1:06:13 | 1:06:16 | guest seated alone on the couch facing the camera |
| `20220616_Dv_-QKO2E1M.mp4` | 1:11:07 | 1:11:10 | guest seated alone on the couch facing the camera |

## Dropped as still figures

23 sections were dropped automatically because the figure's pixels never changed (statues, posters, artwork). They are listed in `standing/output/dropped.csv`; all were checked and none was a real person.
